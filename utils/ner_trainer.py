import torch
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from seqeval.metrics import classification_report as classification_report_seqeval
from sklearn.metrics import classification_report as classification_report_sklearn

from sklearn.metrics import f1_score as f1_score_sklearn
from apex import amp
from transformers import AdamW
from tqdm import tqdm_notebook as tqdm

from utils import utils


class NERTrainer(object):
    """ Trainer of BERT model """

    def __init__(self,
                 model,
                 train_dataloader,
                 valid_dataloader,
                 label_list,
                 fp16=False):
        """
        :param model:            [transformers BertForTokenClassification]
        :param train_dataloader: [pytorch DataLoader]
        :param valid_dataloader: [pytorch DataLoader]
        :param label_list:       [list] of [str]
        :param fp16:             [bool]
        """
        # input attributes
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.label_list = label_list
        self.fp16 = fp16

        # derived attributes
        self.label_ids_filtered = [self.label_list.index(label)
                                   for label in self.label_list
                                   if not (label.startswith('[') or label == 'O')]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # if fp16:
        #     self.model.half()

        self.writer = SummaryWriter()

        """
        self.loss_fct = CrossEntropyLoss(ignore_index=0)  # NEW: HERE 3
        # self.loss_fct = CrossEntropyLoss()  # OLD: HERE 3
        """

        # no input arguments
        # learning rate
        self.learning_rate = None
        self.warmup_proportion = None
        self.optimizer = None

        self.total_steps = None

        # metrics
        self.metrics = dict()
        for elem1 in ['batch', 'epoch']:
            self.metrics[elem1] = dict()
            for elem2 in ['train', 'valid']:
                self.metrics[elem1][elem2] = dict()
                for elem3 in ['loss', 'acc']:
                    self.metrics[elem1][elem2][elem3] = list()
                for elem3 in ['f1']:
                    self.metrics[elem1][elem2][elem3] = dict()
                    for elem4 in ['macro', 'micro']:
                        self.metrics[elem1][elem2][elem3][elem4] = dict()
                        for elem5 in ['all', 'fil']:
                            self.metrics[elem1][elem2][elem3][elem4][elem5] = list()
        self.metrics['batch']['train']['lr'] = list()

    ####################################################################################################################
    # 1. FIT & VALIDATION
    ####################################################################################################################
    def fit(self,
            num_epochs=25,
            max_grad_norm=2.0,
            learning_rate=3e-5,
            warmup_proportion=0.1,
            verbose=False):

        # learning rate
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion

        # optimizer
        self.optimizer = self.create_optimizer(self.learning_rate, self.fp16)
        if self.device == "cpu":
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        # total_steps & global_step
        self.total_steps = self.get_total_steps(num_epochs)
        global_step = None

        ################################################################################################################
        # start training
        ################################################################################################################
        pbar = tqdm(total=num_epochs*len(self.train_dataloader))
        self.model.zero_grad()  ## HERE 2
        self.model.train()

        for epoch in range(num_epochs):
            print("\n>>> Train Epoch: {}".format(epoch))

            for step, batch in enumerate(self.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                if verbose and step == 0:
                    print('\n--- input tensors ---')
                    print('> input_ids[0]:', input_ids.shape)
                    print(input_ids[0])
                    print('> input_mask[0]:', input_mask.shape)
                    print(input_mask[0])
                    print('> segment_ids[0]:', segment_ids.shape)
                    print(segment_ids[0])
                    print('> label_ids[0]:', label_ids.shape)
                    print(label_ids[0])

                """
                logits = self.model(input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids,

                                    # OLD: HERE 4a
                                    # attention_mask=segment_ids,
                                    # token_type_ids=input_mask,
                                    )
                loss = self.loss(logits, label_ids)
                """
                # NEW: HERE 5
                outputs = self.model(input_ids,
                                     attention_mask=input_mask,
                                     token_type_ids=segment_ids,
                                     labels=label_ids,
                                     )
                batch_train_loss, logits = outputs[:2]

                # batch loss
                if verbose:
                    print(f'Batch #{step} train loss: {batch_train_loss:.2f}')

                # to cpu/numpy
                np_batch_train_loss = batch_train_loss.detach().cpu().numpy()
                np_logits = logits.detach().cpu().numpy()     # shape: [batch_size, seq_length, num_labels]
                np_label_ids = label_ids.to('cpu').numpy()    # shape: [batch_size, seq_legnth]

                # batch train metrics
                batch_train_metrics, _, progress_bar = \
                    self.compute_batch_metrics(np_batch_train_loss, np_logits, np_label_ids)

                # training progressbar
                pbar.update(1)  # TQDM - progressbar
                pbar.set_description(progress_bar)

                # update learning rate
                global_step = self.get_global_step(epoch, step)
                lr_this_step = self.update_learning_rate(global_step)
                self.metrics['batch']['train']['lr'].append(lr_this_step)
                if verbose:
                    print(f'          learning rate: {lr_this_step:.2e}')

                # backpropagation
                if self.fp16:
                    with amp.scale_loss(batch_train_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    batch_train_loss.backward()

                # TODO undersök varför man vill göra det här, det får ibland modellen att inte lära sig
                # self.clip_grad_norm(max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()

                # writer
                self.writer.add_scalar('train/loss', batch_train_metrics['loss'], global_step)
                self.writer.add_scalar('train/acc', batch_train_metrics['acc'], global_step)
                self.writer.add_scalar('train/f1_macro_all', batch_train_metrics['f1_macro_all'], global_step)
                self.writer.add_scalar('train/learning_rate', lr_this_step, global_step)

            self.validation(global_step, epoch)

    def validation(self, global_step, epoch, verbose=False):
        print("\n>>> Valid Epoch: {}".format(epoch))
        self.model.eval()
        epoch_valid_metrics = {
            'loss': 0,
            'acc': 0,
        }
        nb_eval_steps, nb_eval_examples = 0, 0
        epoch_valid_pred_label_ids = list()
        epoch_valid_true_label_ids = list()

        for batch in self.valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            with torch.no_grad():
                outputs = self.model(input_ids,
                                     attention_mask=input_mask,
                                     token_type_ids=segment_ids,
                                     labels=label_ids)
                """
                logits = self.model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    token_type_ids=b_segment_ids,

                                    # OLD: HERE 4b
                                    # attention_mask=b_segment_ids,
                                    # token_type_ids=b_input_mask,
                                    )
                """
                batch_valid_loss, logits = outputs[:2]

            # to cpu/numpy
            np_batch_valid_loss = batch_valid_loss.detach().cpu().numpy()
            np_logits = logits.detach().cpu().numpy()
            np_label_ids = label_ids.to('cpu').numpy()

            # batch valid metrics
            batch_valid_metrics, batch_valid_label_ids, _ = \
                self.compute_batch_metrics(np_batch_valid_loss, np_logits, np_label_ids)

            # epoch
            epoch_valid_pred_label_ids.extend(batch_valid_label_ids['pred'])
            epoch_valid_true_label_ids.extend(batch_valid_label_ids['true'])

            # epoch metrics
            epoch_valid_metrics['loss'] += batch_valid_metrics['loss']
            epoch_valid_metrics['acc'] += batch_valid_metrics['acc']

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        epoch_valid_metrics['loss'] = epoch_valid_metrics['loss']/nb_eval_steps
        epoch_valid_metrics['acc'] = epoch_valid_metrics['acc']/nb_eval_steps

        ##########################
        # compute f1 score
        ##########################
        # all
        epoch_valid_metrics['f1_macro_all'], epoch_valid_metrics['f1_micro_all'] = \
            self.f1_score(epoch_valid_pred_label_ids, epoch_valid_true_label_ids)

        # fil
        epoch_valid_metrics['f1_macro_fil'], epoch_valid_metrics['f1_micro_fil'] = \
            self.f1_score(epoch_valid_pred_label_ids, epoch_valid_true_label_ids, filtered_label_ids=True)

        ##########################
        # report
        ##########################
        self.metrics['epoch']['valid']['loss'].append(epoch_valid_metrics['loss'])
        self.metrics['epoch']['valid']['acc'].append(epoch_valid_metrics['acc'])
        self.metrics['epoch']['valid']['f1']['macro']['all'].append(epoch_valid_metrics['f1_macro_all'])
        self.metrics['epoch']['valid']['f1']['micro']['all'].append(epoch_valid_metrics['f1_micro_all'])
        self.metrics['epoch']['valid']['f1']['macro']['fil'].append(epoch_valid_metrics['f1_macro_fil'])
        self.metrics['epoch']['valid']['f1']['micro']['fil'].append(epoch_valid_metrics['f1_micro_fil'])

        print('Epoch #{} valid loss:              {:.2f}'.format(epoch, epoch_valid_metrics['loss']))
        print('Epoch #{} valid acc:               {:.2f}'.format(epoch, epoch_valid_metrics['acc']))
        print('Epoch #{} valid f1 (macro, all):   {:.2f}'.format(epoch, epoch_valid_metrics['f1_macro_all']))
        print('Epoch #{} valid f1 (micro, all):   {:.2f}'.format(epoch, epoch_valid_metrics['f1_micro_all']))
        print('Epoch #{} valid f1 (macro, fil):   {:.2f}'.format(epoch, epoch_valid_metrics['f1_macro_fil']))
        print('Epoch #{} valid f1 (micro, fil):   {:.2f}'.format(epoch, epoch_valid_metrics['f1_micro_fil']))

        # writer
        self.writer.add_scalar('validation/loss', epoch_valid_metrics['loss'], global_step)
        self.writer.add_scalar('validation/acc', epoch_valid_metrics['acc'], global_step)
        self.writer.add_scalar('validation/f1_score_macro_all', epoch_valid_metrics['f1_macro_all'], global_step)
        self.writer.add_scalar('validation/f1_score_micro_all', epoch_valid_metrics['f1_micro_all'], global_step)

        ##########################
        # classification reports
        ##########################
        # use labels instead of label_ids
        epoch_valid_pred_labels = [self.label_list[label_id] for label_id in epoch_valid_pred_label_ids]
        epoch_valid_true_labels = [self.label_list[label_id] for label_id in epoch_valid_true_label_ids]

        # token-based classification report
        print('\n--- token-based (sklearn) classification report ---')
        print(classification_report_sklearn(epoch_valid_pred_labels,
                                            epoch_valid_true_labels,
                                            labels=[label for label in self.label_list if label != 'O']))

        # enrich pred_tags & valid_tags with bio prefixes
        epoch_valid_pred_labels_bio = utils.add_bio_to_label_list(epoch_valid_pred_labels)
        epoch_valid_true_labels_bio = utils.add_bio_to_label_list(epoch_valid_true_labels)
        if verbose:
            print('\n--- predicted and true labels ---')
            print(epoch_valid_pred_labels[:10], len(epoch_valid_pred_labels))
            print(epoch_valid_true_labels[:10], len(epoch_valid_true_labels))
            print(epoch_valid_pred_labels_bio[:10], len(epoch_valid_pred_labels_bio))
            print(epoch_valid_true_labels_bio[:10], len(epoch_valid_true_labels_bio))

        # chunk-based classification report
        print('\n--- chunk-based (seqeval) classification report ---')
        print(classification_report_seqeval(epoch_valid_true_labels_bio,
                                            epoch_valid_pred_labels_bio,
                                            suffix=False))

    ####################################################################################################################
    # 2. METRICS
    ####################################################################################################################
    @staticmethod
    def reduce_and_flatten(_np_logits, _np_label_ids):
        """
        reduce _np_logits (3D -> 2D), flatten both np arrays (2D -> 1D)
        ---------------------------------------------------------------
        :param _np_logits:    [np array] of shape [batch_size, seq_length, num_labels]
        :param _np_label_ids: [np array] of shape [batch_size, seq_length]
        :return: pred_flat:   [np array] of shape [batch_size * seq_length], _np_logits    reduced and flattened
                 true_flat:   [np array] of shape [batch_size * seq_length], _np_label_ids             flattened
        """
        pred_flat = np.argmax(_np_logits, axis=2).flatten()
        true_flat = _np_label_ids.flatten()
        return pred_flat, true_flat

    def compute_batch_metrics(self, np_batch_train_loss, np_logits, np_label_ids):
        metrics = {'loss': np_batch_train_loss}

        # batch
        label_ids = dict()
        label_ids['pred'], label_ids['true'] = \
            self.reduce_and_flatten(np_logits, np_label_ids)  # shape: batch_size * seq_length

        # batch metrics
        metrics['acc'] = self.accuracy(label_ids['pred'], label_ids['true'])
        metrics['f1_macro_all'], metrics['f1_micro_all'] = \
            self.f1_score(label_ids['pred'], label_ids['true'])
        metrics['f1_macro_fil'], metrics['f1_micro_fil'] = \
            self.f1_score(label_ids['pred'], label_ids['true'], filtered_label_ids=True)

        # append to self.metrics
        self.metrics['batch']['train']['loss'].append(metrics['loss'])
        self.metrics['batch']['train']['acc'].append(metrics['acc'])
        self.metrics['batch']['train']['f1']['macro']['all'].append(metrics['f1_macro_all'])
        self.metrics['batch']['train']['f1']['micro']['all'].append(metrics['f1_micro_all'])
        self.metrics['batch']['train']['f1']['macro']['fil'].append(metrics['f1_macro_fil'])
        self.metrics['batch']['train']['f1']['micro']['fil'].append(metrics['f1_micro_fil'])

        # progress bar
        _progress_bar = 'acc: {:.2f} | f1 (macro, all): {:.2f} | f1 (micro, all): {:.2f}'.format(
            metrics['acc'],
            metrics['f1_macro_all'],
            metrics['f1_micro_all'],
        )

        return metrics, label_ids, _progress_bar

    @staticmethod
    def accuracy(_pred_flat, _true_flat):
        """
        computes accuracy of predictions (_np_logits) w.r.t. ground truth (_np_label_ids)
        ---------------------------------------------------------------------------------
        :param _pred_flat:   [np array] of shape [batch_size * seq_length]
        :param _true_flat:   [np array] of shape [batch_size * seq_length]
        :return: acc [np float]
        """
        return np.sum(_pred_flat == _true_flat) / len(_true_flat)

    def f1_score(self, _pred_flat, _true_flat, filtered_label_ids=False):
        """
        computes f1 score (macro/micro) of predictions (_pred_flat) w.r.t. ground truth (_true_flat)
        -----------------------------------------------------------------------------------------------
        :param _pred_flat:   [np array] of shape [batch_size * seq_length]
        :param _true_flat:   [np array] of shape [batch_size * seq_length]
        :param filtered_label_ids: [bool] if True, filter label_ids such that only relevant label_ids remain
        :return: f1_score_macro [np array] f1 score for each class, then averaged
                 f1_score_micro [np array] f1 score for all examples
        """
        # filter label_ids
        labels = self.label_ids_filtered if filtered_label_ids else None

        # compute f1 scores
        return f1_score_sklearn(_pred_flat, _true_flat, labels=labels, average='macro'), \
            f1_score_sklearn(_pred_flat, _true_flat, labels=labels, average='micro')

    ####################################################################################################################
    # 3. STEPS
    ####################################################################################################################
    def get_global_step(self, epoch, step):
        return epoch * len(self.train_dataloader) + step
    
    def get_total_steps(self, num_epochs):
        """
        gets total_steps = num_epochs * (number of training data samples)
        -----------------------------------------------------------------
        :param num_epochs:    [int], e.g. 10
        :return: total_steps: [int], e.g. 2500 (in case of 250 training data samples)
        """
        return num_epochs * len(self.train_dataloader)

    ####################################################################################################################
    # 4. OPTIMIZER
    ####################################################################################################################
    def clip_grad_norm(self, max_grad_norm):
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
        
    def update_learning_rate(self, global_step):
        lr_this_step = self.learning_rate * self.warmup_linear((global_step + 1) / self.total_steps,
                                                               self.warmup_proportion)
        """
        print('> global_step:', global_step)
        print('> self.learning_rate:', self.learning_rate)
        print('> self.total_steps:', self.total_steps)
        print('> (global_step+1)/self.total_steps:', (global_step + 1) / self.total_steps)
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step
        return lr_this_step

    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - (x - warmup)/(1 - warmup)
    
    def create_optimizer(self, learning_rate, fp16=True, no_decay=['bias', 'gamma', 'beta']):
        # Remove unused pooler that otherwise break Apex
        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.02},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        if fp16:
            # optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.learning_rate, bias_correction=False)
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
            # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            
        else:
            optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)
            # optimizer = FusedAdam(optimizer_grouped_parameters, lr=learning_rate)
        
        # optimizer = BertAdam(optimizer_grouped_parameters,lr=2e-5, warmup=.1)
        return optimizer

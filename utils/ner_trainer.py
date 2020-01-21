import torch
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from seqeval.metrics import f1_score as f1_score_seqeval
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

            # for step, batch in enumerate(progress_bar(self.train_dataloader, parent=epoch_process)):
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
                batch_train_loss, logits_tmp = outputs[:2]
                logits = [logits_tmp]
                if verbose:
                    print(f'Batch #{step} train loss: {batch_train_loss:.2f}')

                # to cpu/numpy
                np_logits = logits[0].detach().cpu().numpy()
                np_label_ids = label_ids.to('cpu').numpy()

                # metrics
                batch_train_acc = self.accuracy(np_logits, np_label_ids)
                batch_train_f1_macro_all, batch_train_f1_micro_all = \
                    self.f1_score(np_logits, np_label_ids, flatten=True, filtered_label_ids=False)
                batch_train_f1_macro_fil, batch_train_f1_micro_fil = \
                    self.f1_score(np_logits, np_label_ids, flatten=True, filtered_label_ids=True, label_str=False)
                batch_train_loss_mean = batch_train_loss.mean().item()
                self.metrics['batch']['train']['loss'].append(batch_train_loss_mean)
                self.metrics['batch']['train']['acc'].append(batch_train_acc)
                self.metrics['batch']['train']['f1']['macro']['all'].append(batch_train_f1_macro_all)
                self.metrics['batch']['train']['f1']['micro']['all'].append(batch_train_f1_micro_all)
                self.metrics['batch']['train']['f1']['macro']['fil'].append(batch_train_f1_macro_fil)
                self.metrics['batch']['train']['f1']['micro']['fil'].append(batch_train_f1_micro_fil)
                pbar.update(1)  # TQDM - progressbar
                description_str = f'acc: {batch_train_acc:.2f} | ' + \
                                  f'f1 (macro, all): {batch_train_f1_macro_all:.2f} | ' + \
                                  f'f1 (micro, all): {batch_train_f1_micro_all:.2f}'
                pbar.set_description(description_str)

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
                self.writer.add_scalar('train/loss', batch_train_loss_mean, global_step)
                self.writer.add_scalar('train/acc', batch_train_acc, global_step)
                self.writer.add_scalar('train/f1_macro_all', batch_train_f1_macro_all, global_step)
                self.writer.add_scalar('train/learning_rate', lr_this_step, global_step)

            self.validation(global_step, epoch)
                
    def validation(self, global_step, epoch, verbose=False):
        print("\n>>> Valid Epoch: {}".format(epoch))
        self.model.eval()
        epoch_valid_loss, epoch_valid_acc = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred_label_ids, true_label_ids = [], []
        
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
                batch_valid_loss, logits_tmp = outputs[:2]
                logits = [logits_tmp]

            # to cpu/numpy
            np_logits = logits[0].detach().cpu().numpy()
            np_label_ids = label_ids.to('cpu').numpy()

            pred_label_ids.extend([list(p) for p in np.argmax(np_logits, axis=2)])
            true_label_ids.append(np_label_ids)

            batch_valid_loss_mean = batch_valid_loss.mean().item()
            batch_valid_acc = self.accuracy(np_logits, np_label_ids)
            batch_valid_f1_macro_all, batch_valid_f1_micro_all = \
                self.f1_score(np_logits, np_label_ids, flatten=True, filtered_label_ids=False)
            batch_valid_f1_macro_fil, batch_valid_f1_micro_fil = \
                self.f1_score(np_logits, np_label_ids, flatten=True, filtered_label_ids=True, label_str=False)

            self.metrics['batch']['valid']['loss'].append(batch_valid_loss_mean)
            self.metrics['batch']['valid']['acc'].append(batch_valid_acc)
            self.metrics['batch']['valid']['f1']['macro']['all'].append(batch_valid_f1_macro_all)
            self.metrics['batch']['valid']['f1']['micro']['all'].append(batch_valid_f1_micro_all)
            self.metrics['batch']['valid']['f1']['macro']['fil'].append(batch_valid_f1_macro_fil)
            self.metrics['batch']['valid']['f1']['micro']['fil'].append(batch_valid_f1_micro_fil)

            epoch_valid_loss += batch_valid_loss_mean
            epoch_valid_acc += batch_valid_acc

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        epoch_valid_loss = epoch_valid_loss/nb_eval_steps
        epoch_valid_acc = epoch_valid_acc/nb_eval_steps

        ##########################
        # flatten pred & true
        ##########################
        # turn nested label_ids (=[[0, 1, 2], [3, 4, 5]]) into flat labels (=['O', 'ORG', .., 'PER'])
        pred_labels_flat_all = [self.label_list[p_i] for p in pred_label_ids for p_i in p]
        true_labels_flat_all = [self.label_list[l_ii] for l in true_label_ids for l_i in l for l_ii in l_i]

        ##########################
        # compute f1 score
        ##########################
        # all
        epoch_valid_f1_macro_all, epoch_valid_f1_micro_all = \
            self.f1_score(pred_labels_flat_all, true_labels_flat_all,
                          flatten=False, filtered_label_ids=False)

        # fil
        epoch_valid_f1_macro_fil, epoch_valid_f1_micro_fil = \
            self.f1_score(pred_labels_flat_all, true_labels_flat_all,
                          flatten=False, filtered_label_ids=True, label_str=True)

        ##########################
        # report
        ##########################
        self.metrics['epoch']['valid']['loss'].append(epoch_valid_loss)
        self.metrics['epoch']['valid']['acc'].append(epoch_valid_acc)
        self.metrics['epoch']['valid']['f1']['macro']['all'].append(epoch_valid_f1_macro_all)
        self.metrics['epoch']['valid']['f1']['micro']['all'].append(epoch_valid_f1_micro_all)
        self.metrics['epoch']['valid']['f1']['macro']['fil'].append(epoch_valid_f1_macro_fil)
        self.metrics['epoch']['valid']['f1']['micro']['fil'].append(epoch_valid_f1_micro_fil)

        print(f'Epoch #{epoch} valid loss:              {epoch_valid_loss:.2f}')
        print(f'Epoch #{epoch} valid acc:               {epoch_valid_acc:.2f}')
        print(f'Epoch #{epoch} valid f1 (macro, all):   {epoch_valid_f1_macro_all:.2f}')
        print(f'Epoch #{epoch} valid f1 (micro, all):   {epoch_valid_f1_micro_all:.2f}')
        print(f'Epoch #{epoch} valid f1 (macro, fil):   {epoch_valid_f1_macro_fil:.2f}')
        print(f'Epoch #{epoch} valid f1 (micro, fil):   {epoch_valid_f1_micro_fil:.2f}')

        print('\n--- token-based (sklearn) classification report ---')
        print(classification_report_sklearn(true_labels_flat_all,
                                            pred_labels_flat_all,
                                            labels=[label for label in self.label_list if label != 'O']))

        # writer
        self.writer.add_scalar('validation/loss', epoch_valid_loss, global_step)
        self.writer.add_scalar('validation/acc', epoch_valid_acc, global_step)
        self.writer.add_scalar('validation/f1_score_macro_all', epoch_valid_f1_macro_all, global_step)
        self.writer.add_scalar('validation/f1_score_micro_all', epoch_valid_f1_micro_all, global_step)

        ##########################
        # BIO-prefixes and chunk-based metrics
        ##########################
        # enrich pred_tags & valid_tags with bio prefixes
        pred_labels_flat_all_bio = utils.add_bio_to_label_list(pred_labels_flat_all)
        true_labels_flat_all_bio = utils.add_bio_to_label_list(true_labels_flat_all)
        if verbose:
            print('\n--- predicted and true labels ---')
            print(pred_labels_flat_all[:10], len(pred_labels_flat_all))
            print(true_labels_flat_all[:10], len(true_labels_flat_all))
            print(pred_labels_flat_all_bio[:10], len(pred_labels_flat_all_bio))
            print(true_labels_flat_all_bio[:10], len(true_labels_flat_all_bio))

        print('\n--- chunk-based (seqeval) classification report ---')
        print(classification_report_seqeval(true_labels_flat_all_bio,
                                            pred_labels_flat_all_bio,
                                            suffix=False))

    ####################################################################################################################
    # 2. METRICS
    ####################################################################################################################
    @staticmethod
    def accuracy(_np_logits, _np_label_ids, flatten=True):
        if flatten:
            pred_flat = np.argmax(_np_logits, axis=2).flatten()
            labels_flat = _np_label_ids.flatten()
        else:
            pred_flat = _np_logits
            labels_flat = _np_label_ids

        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def f1_score(self,
                 _np_logits,
                 _np_label_ids,
                 flatten=True,
                 filtered_label_ids=False,
                 label_str=False):

        labels = self.label_ids_filtered if filtered_label_ids else None
        if filtered_label_ids and label_str:
            labels = [self.label_list[i] for i in labels]

        if flatten:
            pred_flat = np.argmax(_np_logits, axis=2).flatten()
            labels_flat = _np_label_ids.flatten()
        else:
            pred_flat = _np_logits
            labels_flat = _np_label_ids

        return f1_score_sklearn(pred_flat, labels_flat, labels=labels, average='macro'), \
            f1_score_sklearn(pred_flat, labels_flat, labels=labels, average='micro')

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

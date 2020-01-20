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
        self.metrics = {
            'batch': {
                'train': {
                    'loss': list(),
                    'acc': list(),
                    'f1_macro': list(),
                    'f1_micro': list(),
                    'lr': list(),
                },
                'valid': {
                    'loss': list(),
                    'acc': list(),
                    'f1_macro': list(),
                    'f1_micro': list(),
                },
            },
            'epoch': {
                'valid': {
                    'loss': list(),
                    'acc': list(),
                    'f1_macro': list(),
                    'f1_micro': list(),
                },
            },
        }

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
                print(f'Batch #{step} train loss: {batch_train_loss:.2f}')

                # to cpu/numpy
                np_logits = logits[0].detach().cpu().numpy()
                np_label_ids = label_ids.to('cpu').numpy()

                # metrics
                batch_train_acc = self.accuracy(np_logits, np_label_ids)
                batch_train_f1_macro, batch_train_f1_micro = self.f1_score(np_logits, np_label_ids)
                batch_train_loss_mean = batch_train_loss.mean().item()
                self.metrics['batch']['train']['loss'].append(batch_train_loss_mean)
                self.metrics['batch']['train']['acc'].append(batch_train_acc)
                self.metrics['batch']['train']['f1_macro'].append(batch_train_f1_macro)
                self.metrics['batch']['train']['f1_micro'].append(batch_train_f1_micro)
                pbar.update(1)  # TQDM - progressbar
                description_str = f'acc: {batch_train_acc:.2f} | ' + \
                                  f'f1 (macro): {batch_train_f1_macro:.2f} | ' + \
                                  f'f1 (micro): {batch_train_f1_micro:.2f}'
                pbar.set_description(description_str)

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

                # update learning rate
                global_step = self.get_global_step(epoch, step)
                lr_this_step = self.update_learning_rate(global_step)
                print(f'        update learning rate: {lr_this_step:.2e}')
                self.metrics['batch']['train']['lr'].append(lr_this_step)

                # writer
                self.writer.add_scalar('train/loss', batch_train_loss_mean, global_step)
                self.writer.add_scalar('train/acc', batch_train_acc, global_step)
                self.writer.add_scalar('train/f1_macro', batch_train_f1_macro, global_step)
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
            batch_valid_f1_macro, batch_valid_f1_micro = self.f1_score(np_logits, np_label_ids)

            self.metrics['batch']['valid']['loss'].append(batch_valid_loss_mean)
            self.metrics['batch']['valid']['acc'].append(batch_valid_acc)
            self.metrics['batch']['valid']['f1_macro'].append(batch_valid_f1_macro)
            self.metrics['batch']['valid']['f1_micro'].append(batch_valid_f1_micro)

            epoch_valid_loss += batch_valid_loss_mean
            epoch_valid_acc += batch_valid_acc

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        epoch_valid_loss = epoch_valid_loss/nb_eval_steps
        epoch_valid_acc = epoch_valid_acc/nb_eval_steps

        # turn nested label_ids (=[[0, 1, 2], [3, 4, 5]]) into flat labels (=['O', 'ORG', .., 'PER'])
        pred_labels_flat = [self.label_list[p_i] for p in pred_label_ids for p_i in p]
        true_labels_flat = [self.label_list[l_ii] for l in true_label_ids for l_i in l for l_ii in l_i]

        # enrich pred_tags & valid_tags
        pred_labels_flat_bio = utils.add_bio_to_label_list(pred_labels_flat)
        true_labels_flat_bio = utils.add_bio_to_label_list(true_labels_flat)

        if verbose:
            print('\n--- predicted and true labels ---')
            print(pred_labels_flat[:10], len(pred_labels_flat))
            print(true_labels_flat[:10], len(true_labels_flat))
            print(pred_labels_flat_bio[:10], len(pred_labels_flat_bio))
            print(true_labels_flat_bio[:10], len(true_labels_flat_bio))

        # (true_labels_flat, pred_labels_flat) = self.stripTags(true_labels_flat, pred_labels_flat)
        epoch_valid_f1_macro = f1_score_sklearn(pred_labels_flat_bio, true_labels_flat_bio, average='macro')
        epoch_valid_f1_micro = f1_score_sklearn(pred_labels_flat_bio, true_labels_flat_bio, average='micro')

        self.metrics['epoch']['valid']['loss'].append(epoch_valid_loss)
        self.metrics['epoch']['valid']['acc'].append(epoch_valid_acc)
        self.metrics['epoch']['valid']['f1_macro'].append(epoch_valid_f1_macro)
        self.metrics['epoch']['valid']['f1_micro'].append(epoch_valid_f1_micro)

        print(f'Epoch #{epoch} valid loss:       {epoch_valid_loss:.2f}')
        print(f'Epoch #{epoch} valid acc:        {epoch_valid_acc:.2f}')
        print(f'Epoch #{epoch} valid f1 (macro): {epoch_valid_f1_macro:.2f}')
        print(f'Epoch #{epoch} valid f1 (micro): {epoch_valid_f1_micro:.2f}')

        print('\n--- chunk-based (seqeval) classification report ---')
        print(classification_report_seqeval(true_labels_flat_bio,
                                            pred_labels_flat_bio,
                                            suffix=False))

        print('\n--- token-based (sklearn) classification report ---')
        print(classification_report_sklearn(true_labels_flat,
                                            pred_labels_flat,
                                            labels=[label for label in self.label_list if label != 'O']))

        # writer
        self.writer.add_scalar('validation/loss', epoch_valid_loss, global_step)
        self.writer.add_scalar('validation/acc', epoch_valid_acc, global_step)
        self.writer.add_scalar('validation/f1_score_macro', epoch_valid_f1_macro, global_step)
        self.writer.add_scalar('validation/f1_score_micro', epoch_valid_f1_micro, global_step)

    ####################################################################################################################
    # 2. METRICS
    ####################################################################################################################
    """
    def loss(self, logits, label_ids):
        loss = self.loss_fct(logits[0].view(-1, self.model.num_labels), label_ids.view(-1))
        return loss
    """

    @staticmethod
    def accuracy(_np_logits, _np_label_ids):
        pred_flat = np.argmax(_np_logits, axis=2).flatten()
        labels_flat = _np_label_ids.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def f1_score(_np_logits, _np_label_ids):
        pred_flat = np.argmax(_np_logits, axis=2).flatten()
        labels_flat = _np_label_ids.flatten()
        # (labels_flat, pred_flat) = self.stripTags(labels_flat, pred_flat)
        return f1_score_sklearn(pred_flat, labels_flat, average='macro'), \
            f1_score_sklearn(pred_flat, labels_flat, average='micro')

    """
    def f1_score_default_accuracy(self, _np_logits, _np_label_ids):
        predictions , true_labels = [], []
        predictions.extend([list(p) for p in np.argmax(_np_logits, axis=2)])
        true_labels.append(_np_label_ids)
        pred_tags = [self.label_list[p_i] for p in predictions for p_i in p]
        valid_tags = [self.label_list[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        if self.global_step == 0:
            print(_np_logits.shape, _np_label_ids.shape)
        return f1_score_seqeval(pred_tags, valid_tags)
        
    def stripTags(self, labels_flat, pred_flat):
        # Only calculate F1-score on interesting labels.
        for idx, lbl in enumerate(labels_flat):
            if lbl == 'O' or lbl == '[PAD]' or lbl == '[CLS]' or lbl == '[SEP]':
                pred_flat.pop(idx)
                labels_flat.pop(idx)
        return labels_flat, pred_flat
    """

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
        torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm=max_grad_norm)
        
    def update_learning_rate(self, global_step):
        lr_this_step = self.learning_rate * self.warmup_linear(global_step / self.total_steps, self.warmup_proportion)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step
        return lr_this_step

    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x
    
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

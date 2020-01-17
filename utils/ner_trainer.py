import torch
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from seqeval.metrics import f1_score as f1_score_seqeval
from seqeval.metrics import classification_report

from sklearn.metrics import f1_score as f1_score_sklearn
from apex import amp
from transformers import AdamW
from tqdm import tqdm_notebook as tqdm


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
                    'f1': list(),
                    'lr': list(),
                },
                'valid': {
                    'loss': list(),
                    'acc': list(),
                    'f1': list(),
                },
            },
            'epoch': {
                'valid': {
                    'loss': list(),
                    'acc': list(),
                    'f1': list(),
                },
            },
        }
        self.labelDict = None

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

        self.optimizer = self.create_optimizer(self.learning_rate, self.fp16)
        if self.device == "cpu":
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        self.total_steps = self.get_total_steps(num_epochs)

        # metrics
        """
        self.accuracy_hist = np.array([])
        self.f1_score_hist = np.array([])
        self.loss_hist = np.array([])
        self.val_f1_score_hist = np.array([])
        self.val_accuracy_hist = np.array([])
        self.val_loss_hist = np.array([])
        """
        self.labelDict = {
            k: {'total': 0, 'correct': 0}
            for k in self.label_list
        }
        
        # epoch_process = master_bar(range(num_epochs))
        # for epoch in epoch_process:
        pbar = tqdm(total=num_epochs*len(self.train_dataloader))
        global_step = None
        self.model.zero_grad()  ## HERE 2

        for epoch in range(num_epochs):
            print()
            print(">>> Train Epoch: {}".format(epoch))
            self.model.train()
            
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
                loss, logits_tmp = outputs[:2]
                logits = [logits_tmp]
                print(f'Batch #{step} train loss: {loss}')

                # metrics
                acc = self.accuracy(logits, label_ids)
                f1 = self.f1_score_accuracy(logits, label_ids)
                self.metrics['batch']['train']['loss'].append(loss.mean().item())
                self.metrics['batch']['train']['acc'].append(acc)
                self.metrics['batch']['train']['f1'].append(f1)
                pbar.update(1)  # TQDM - progressbar
                pbar.set_description("Acc: " + str(round(acc, 5)) + " F1: " + str(round(f1, 5)))

                # backpropagation
                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # TODO undersök varför man vill göra det här, det får ibland modellen att inte lära sig
                # self.clip_grad_norm(max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()

                # update learning rate
                global_step = self.get_global_step(epoch, step)
                lr_this_step = self.update_learning_rate(global_step)
                print(f'        update learning rate: {lr_this_step}')
                self.metrics['batch']['train']['lr'].append(lr_this_step)

                # writer
                self.writer.add_scalar('train/accuracy', acc, global_step)
                self.writer.add_scalar('train/f1_score', f1, global_step)
                self.writer.add_scalar('train/loss', loss.mean().item(), global_step)
                self.writer.add_scalar('train/learning_rate', lr_this_step, global_step)

            self.validation(global_step, epoch)
                
    def validation(self, global_step, epoch, verbose=False):
        print()
        print(">>> Valid Epoch: {}".format(epoch))
        self.model.eval()
        epoch_valid_loss, epoch_valid_acc = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        
        resLabelDict = {
            k: {'total': 0, 'correct': 0, 'false_positives': 0}
            for k in self.label_list
        }
            
        for batch in self.valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
            
            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                     attention_mask=b_input_mask,
                                     token_type_ids=b_segment_ids,
                                     labels=b_labels)
                """
                logits = self.model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    token_type_ids=b_segment_ids,

                                    # OLD: HERE 4b
                                    # attention_mask=b_segment_ids,
                                    # token_type_ids=b_input_mask,
                                    )
                """
                tmp_eval_loss, logits_tmp = outputs[:2]
                logits = [logits_tmp]
                # print(f'eval_loss: {tmp_eval_loss}')

            f1 = self.f1_score_accuracy(logits, b_labels)
            # print("F1-Score (macro) sklearn: {}".format(f1))
            
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            
            tmp_eval_acc = self.flat_accuracy(logits, label_ids)
            
            epoch_valid_loss += tmp_eval_loss.mean().item()
            epoch_valid_acc += tmp_eval_acc

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

            self.metrics['batch']['valid']['loss'].append(tmp_eval_loss.mean().item())
            self.metrics['batch']['valid']['acc'].append(tmp_eval_acc)
            self.metrics['batch']['valid']['f1'].append(f1)

        epoch_valid_loss = epoch_valid_loss/nb_eval_steps
        epoch_valid_acc = epoch_valid_acc/nb_eval_steps

        pred_tags = [self.label_list[p_i] for p in predictions for p_i in p]
        valid_tags = [self.label_list[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

        # enrich pred_tags & valid_tags
        def enrich(_tag, _previous_tag):
            if _tag == 'O' or _tag.startswith('['):
                return _tag
            elif _previous_tag is None:
                return f'B-{_tag}'
            elif _tag != _previous_tag:
                return f'B-{_tag}'
            else:
                return f'I-{_tag}'

        pred_tags_enriched = [enrich(pred_tags[i], pred_tags[i-1] if i > 0 else None) for i in range(len(pred_tags))]
        valid_tags_enriched = [enrich(valid_tags[i], valid_tags[i-1] if i > 0 else None) for i in range(len(valid_tags))]

        if verbose:
            print('\n--- predicted and true labels ---')
            print(pred_tags[:10], len(pred_tags))
            print(valid_tags[:10], len(valid_tags))
            print(pred_tags_enriched[:10], len(pred_tags_enriched))
            print(valid_tags_enriched[:10], len(valid_tags_enriched))

        # (valid_tags, pred_tags) = self.stripTags(valid_tags, pred_tags)
        epoch_valid_f1 = f1_score_sklearn(pred_tags_enriched, valid_tags_enriched, average='macro')

        self.metrics['epoch']['valid']['loss'].append(epoch_valid_loss)
        self.metrics['epoch']['valid']['acc'].append(epoch_valid_acc)
        self.metrics['epoch']['valid']['f1'].append(epoch_valid_f1)

        print(f'Epoch #{epoch} valid loss: {epoch_valid_loss}')
        print(f'Epoch #{epoch} valid acc:  {epoch_valid_acc}')
        print(f'Epoch #{epoch} valid f1:   {epoch_valid_f1}')

        self.calculate_percentage_correct(pred_tags, valid_tags)

        print('\n--- chunk-based (seqeval) classification report ---')
        print(classification_report(valid_tags_enriched, pred_tags_enriched))

        for idx, tag in enumerate(valid_tags):
            if tag in resLabelDict:
                resLabelDict[tag]['total'] += 1
                if pred_tags[idx] == tag:
                    resLabelDict[tag]['correct'] += 1

        print('\n--- token-based classification report ---')
        count = {'total': 0, 'correct': 0}
        for label in self.label_list:
            if label in resLabelDict.keys():
                print(label, resLabelDict[label])
                for field in count.keys():
                    count[field] += resLabelDict[label][field]
        for field in count.keys():
            c = count[field]
            print(f'{field} = {c}')

        # writer
        self.writer.add_scalar('validation/loss', epoch_valid_loss, global_step)
        self.writer.add_scalar('validation/accuracy', epoch_valid_acc, global_step)
        self.writer.add_scalar('validation/f1_score', epoch_valid_f1, global_step)

    ####################################################################################################################
    # 2. METRICS
    ####################################################################################################################
    def loss(self, logits, label_ids):
        loss = self.loss_fct(logits[0].view(-1, self.model.num_labels), label_ids.view(-1))
        return loss

    def calculate_percentage_correct(self, pred_tags, valid_tags):
        for idx, tag in enumerate(valid_tags):
            if tag in self.labelDict:
                self.labelDict[tag]['total'] += 1
                if pred_tags[idx] == tag:
                    self.labelDict[tag]['correct'] += 1
        
    def f1_score_default_accuracy(self, logits, label_ids):
        predictions , true_labels = [], []
        np_logits = logits[0].detach().cpu().numpy()
        np_label_ids = label_ids.cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(np_logits, axis=2)])
        true_labels.append(np_label_ids)
        pred_tags = [self.label_list[p_i] for p in predictions for p_i in p]
        valid_tags = [self.label_list[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        if self.global_step == 0:
            print(np_logits.shape, np_label_ids.shape)
        return f1_score_seqeval(pred_tags, valid_tags)
        
    def f1_score_accuracy(self, logits, label_ids):
        np_logits = logits[0].detach().cpu().numpy()
        np_label_ids = label_ids.to('cpu').numpy()
        pred_flat = np.argmax(np_logits, axis=2).flatten()
        labels_flat = np_label_ids.flatten()
        # (labels_flat, pred_flat) = self.stripTags(labels_flat, pred_flat)
        return f1_score_sklearn(pred_flat, labels_flat, average='macro')
    
    def stripTags(self, labels_flat, pred_flat):
        # Only calculate F1-score on interesting labels.
        for idx, lbl in enumerate(labels_flat):
            if lbl == 'O' or lbl == '[PAD]' or lbl == '[CLS]' or lbl == '[SEP]':
                pred_flat.pop(idx)
                labels_flat.pop(idx)
        return (labels_flat, pred_flat)
    
    # todo replace with accuracy function
    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def accuracy(logits, label_ids):
        np_logits = logits[0].detach().cpu().numpy()
        np_label_ids = label_ids.to('cpu').numpy()
        pred_flat = np.argmax(np_logits, axis=2).flatten()
        labels_flat = np_label_ids.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

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

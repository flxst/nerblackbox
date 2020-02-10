import logging
import argparse

import torch
from transformers import BertTokenizer, BertForTokenClassification

from utils.utils import preprocess_data
from utils.utils import get_dataset_path
from utils.ner_trainer import NERTrainer

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def main(args):

    hyperparams = {
        'batch_size': args.batch_size,
        'max_seq_length': args.max_seq_length,
        'num_epochs': args.num_epochs,
        'prune_ratio': (args.prune_ratio_train, args.prune_ratio_valid),
        'learning_rate': {
            'lr_max': args.lr_max,
            'lr_schedule': args.lr_schedule,
            'lr_warmup_fraction': args.lr_warmup_fraction,
            'lr_num_cycles': args.lr_num_cycles,
        },
    }
    dataset_path = get_dataset_path(args.dataset_name)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case=False)  # needs to be False !!

    dataloader, label_list = preprocess_data(dataset_path,
                                             tokenizer,
                                             hyperparams['batch_size'],
                                             max_seq_length=hyperparams['max_seq_length'],
                                             prune_ratio=hyperparams['prune_ratio']
                                             )

    model = BertForTokenClassification.from_pretrained(args.pretrained_model_name,
                                                       num_labels=len(label_list))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = NERTrainer(model,
                         train_dataloader=dataloader['train'],
                         valid_dataloader=dataloader['valid'],
                         label_list=label_list,
                         fp16=True if torch.cuda.is_available() else False,
                         verbose=False,
                         )
    trainer.fit(num_epochs=hyperparams['num_epochs'],
                **hyperparams['learning_rate'],
                )

    trainer.save_model_checkpoint(args.dataset_name,
                                  args.pretrained_model_name,
                                  hyperparams['num_epochs'],
                                  hyperparams['prune_ratio'][0],
                                  hyperparams['learning_rate']['lr_schedule'],
                                  )

    trainer.save_metrics(args.dataset_name,
                         args.pretrained_model_name,
                         hyperparams['num_epochs'],
                         hyperparams['prune_ratio'][0],
                         hyperparams['learning_rate']['lr_schedule'],
                         )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type=str, default='af-ai-center/bert-base-swedish-uncased')
    parser.add_argument('--dataset_name', type=str, default='swedish_ner_corpus')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--prune_ratio_train', type=float, default=0.01)
    parser.add_argument('--prune_ratio_valid', type=float, default=0.01)
    parser.add_argument('--lr_max', type=float, default=2e-5)
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--lr_warmup_fraction', type=float, default=0.1)
    parser.add_argument('--lr_num_cycles', type=float, default=None)

    _args = parser.parse_args()
    main(_args)

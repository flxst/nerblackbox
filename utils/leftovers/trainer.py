# from apex.optimizers import FP16_Optimizer
from apex.optimizers import FusedAdam
# from fastprogress import master_bar, progress_bar
from pytorch_pretrained_bert import BertAdam
from torch.optim import Adam


def create_optimizer(model, fp16=True, no_decay=['bias', 'gamma', 'beta']):
    # Remove unused pooler that otherwise break Apex
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.02},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if fp16:
        # optimizer = FusedAdam(optimizer_grouped_parameters, lr=3e-5, bias_correction=False, max_grad_norm=1.0)
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=3e-5)
        # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    return optimizer

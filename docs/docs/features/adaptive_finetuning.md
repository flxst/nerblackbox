# Adaptive Fine-tuning

Adaptive fine-tuning (introduced in [this paper](https://arxiv.org/abs/2202.02617)) is a method that automatically trains for a near-optimal number of epochs.
It is used by default in **nerblackbox**, and corresponds to the 
following [hyperparameters](../../usage/parameters_and_presets/#4-hyperparameters):

??? note "adaptive fine-tuning hyperparameters"
    ``` markdown
    [hparams]
    max_epochs = 250
    early_stopping = True
    monitor = val_loss
    min_delta = 0.0
    patience = 0
    mode = min
    lr_warmup_epochs = 2
    lr_schedule = constant
    lr_cooldown_epochs = 7
    ```

The hyperparameters are also available as a [preset](../../usage/parameters_and_presets/#presets).






from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class CustomEarlyStopping(EarlyStopping):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr_max_epochs = max(self.patience - 1, 0)

    def on_validation_end(self, trainer, pl_module):
        # Original
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return

        self._run_early_stopping_check(trainer)

        # Custom
        use_cooldown = pl_module._hparams.lr_schedule == "constant" and pl_module._hparams.lr_cooldown is True
        if use_cooldown:
            if self.wait_count == 1:  # i.e. NO IMPROVEMENT
                print(f"> Custom Early Stopping: change LR to linear cool-down with {self._lr_max_epochs} epochs.")
                pl_module.scheduler = pl_module._create_scheduler(
                    _lr_warmup_epochs=0,
                    _lr_schedule="linear",
                    _lr_max_epochs=self._lr_max_epochs,
                )
                self.min_delta = -1000000 if self.mode == "min" else 1000000  # => suppress reoccurrence of improvement

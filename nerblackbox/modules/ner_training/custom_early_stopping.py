
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class CustomEarlyStopping(EarlyStopping):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flag_cooldown = False  # Custom
        self._lr_max_epochs = self.patience - 1

    def on_validation_end(self, trainer, pl_module):
        # Original
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return

        self._run_early_stopping_check(trainer)

        # Custom
        use_cooldown = pl_module._hparams.lr_schedule == "constant" and pl_module._hparams.lr_cooldown is True
        if use_cooldown:
            if self.wait_count == 1:
                # NO IMPROVEMENT
                print(f"> Custom Early Stopping: change LR to linear cool-down with {self._lr_max_epochs} epochs.")
                pl_module.scheduler = pl_module._create_scheduler(
                    _lr_warmup_epochs=0,
                    _lr_schedule="linear",
                    _lr_max_epochs=self._lr_max_epochs,
                )
                self.flag_cooldown = True
            elif self.flag_cooldown is True and self.wait_count == 0:
                # REOCCURRENCE OF IMPROVEMENT
                print(f"> Custom Early Stopping: change LR to normal constant mode.")
                pl_module.scheduler = pl_module._create_scheduler(
                    _lr_warmup_epochs=0,
                    _lr_schedule="constant",
                )
                self.flag_cooldown = False

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class CustomEarlyStopping(EarlyStopping):
    def __init__(
        self,
        lr_schedule: str,
        lr_cooldown_epochs: int,
        lr_cooldown_restarts: bool,
        *args,
        **kwargs,
    ):
        self.lr_schedule = lr_schedule
        self.lr_cooldown_epochs = lr_cooldown_epochs
        self.lr_cooldown_restarts = lr_cooldown_restarts
        super().__init__(*args, **kwargs)

        self.use_cooldown = (
            self.lr_schedule == "constant" and self.lr_cooldown_epochs > 0
        )
        # e.g. patience = 4, lr_cooldown_epochs = 3 -> wait_count_trigger_cool_down = 1
        self.wait_count_trigger_cool_down = self.patience - self.lr_cooldown_epochs
        self.flag_cooldown = False  # Custom

    def on_validation_end(self, trainer, pl_module):
        # Original
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return

        self._run_early_stopping_check(trainer)

        # Custom
        if self.use_cooldown:
            if self.wait_count == self.wait_count_trigger_cool_down:
                # NO IMPROVEMENT
                print(
                    f"> Custom Early Stopping: change LR to linear cool-down with {self.lr_cooldown_epochs} epochs."
                )
                pl_module.scheduler = pl_module._create_scheduler(
                    _lr_warmup_epochs=0,
                    _lr_schedule="linear",
                    _lr_max_epochs=self.lr_cooldown_epochs,
                )
                self.flag_cooldown = True
                if self.lr_cooldown_restarts is False:
                    # => suppress reoccurrence of improvement
                    self.min_delta = -1000000 if self.mode == "min" else 1000000
            elif (
                self.lr_cooldown_restarts
                and self.flag_cooldown is True
                and self.wait_count == 0
            ):
                # REOCCURRENCE OF IMPROVEMENT
                print(f"> Custom Early Stopping: change LR to normal constant mode.")
                pl_module.scheduler = pl_module._create_scheduler(
                    _lr_warmup_epochs=0,
                    _lr_schedule="constant",
                )
                self.flag_cooldown = False

from pytorch_lightning.callbacks.base import Callback


class LearningRateChanger(Callback):
    def __init__(self, max_epochs: int, lr_cooldown_epochs: int):
        self.max_epochs = max_epochs
        self.lr_cooldown_epochs = lr_cooldown_epochs
        self.change_epoch = (
            self.max_epochs - self.lr_cooldown_epochs - 1
        )  # minus 1 as epoch counting starts from 0
        super().__init__()

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch == self.change_epoch:
            print(
                f"> LearningRateChanger: change LR to linear cool-down with {self.lr_cooldown_epochs} epochs."
            )
            pl_module.scheduler = pl_module._create_scheduler(
                _lr_warmup_epochs=0,
                _lr_schedule="linear",
                _lr_max_epochs=self.lr_cooldown_epochs,
            )

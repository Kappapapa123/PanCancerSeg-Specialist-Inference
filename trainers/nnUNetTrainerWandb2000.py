"""nnUNet trainer with Weights & Biases logging and 2000 epochs.

Place this file (or symlink it) into the nnUNet trainer variants directory
so that nnUNet can discover it via the -tr flag:

    VARIANTS_DIR=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])")/training/nnUNetTrainer/variants
    ln -sf $(realpath nnUNetTrainerWandb2000.py) "$VARIANTS_DIR/nnUNetTrainerWandb2000.py"

Then train with:
    nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerWandb2000 ...
"""

import os

import torch
import wandb
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandb2000(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000

    def on_train_start(self):
        super().on_train_start()
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "CVPR2026-PanCancerSeg"),
            name=f"{self.plans_manager.dataset_name}_fold{self.fold}",
            config={
                "dataset": self.plans_manager.dataset_name,
                "configuration": self.configuration_name,
                "fold": self.fold,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "patch_size": list(self.configuration_manager.patch_size),
            },
            resume="allow",
        )

    def on_epoch_end(self):
        super().on_epoch_end()

        # Save periodic checkpoint every 200 epochs after epoch 1000
        if self.current_epoch > 1000 and self.current_epoch % 200 == 0:
            self.save_checkpoint(
                os.path.join(self.output_folder, f"checkpoint_epoch{self.current_epoch}.pth")
            )

        logs = self.logger.my_fantastic_logging
        metrics = {"epoch": self.current_epoch}
        if logs["train_losses"]:
            metrics["train_loss"] = logs["train_losses"][-1]
        if logs["val_losses"]:
            metrics["val_loss"] = logs["val_losses"][-1]
        if logs["ema_fg_dice"]:
            metrics["ema_fg_dice"] = logs["ema_fg_dice"][-1]
        if logs["dice_per_class_or_region"]:
            latest = logs["dice_per_class_or_region"][-1]
            for i, d in enumerate(latest):
                metrics[f"dice_class_{i}"] = d
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["epoch_time"] = self.logger.my_fantastic_logging.get(
            "epoch_end_timestamps", [0])[-1] - self.logger.my_fantastic_logging.get(
            "epoch_start_timestamps", [0])[-1]
        wandb.log(metrics, step=self.current_epoch)

    def on_train_end(self):
        super().on_train_end()
        wandb.finish()

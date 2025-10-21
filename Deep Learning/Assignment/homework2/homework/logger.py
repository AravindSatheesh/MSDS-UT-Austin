from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Your code here - finish logging the dummy loss and accuracy

    For training, log the training loss every iteration and the average accuracy every epoch
    Call the loss 'train_loss' and accuracy 'train_accuracy'

    For validation, log only the average accuracy every epoch
    Call the accuracy 'val_accuracy'

    Make sure the logging is in the correct spot so the global_step is set correctly,
    for epoch=0, iteration=0: global_step=0
    """
    # strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # example training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            # log train_loss at current global_step
            logger.add_scalar("train_loss", float(dummy_train_loss), global_step)

            # save train accuracies to be averaged (mean over the 10 samples)
            metrics["train_acc"].append(float(dummy_train_accuracy.mean().item()))

            global_step += 1

        # log average train_accuracy at the end of the epoch
        avg_train_acc = sum(metrics["train_acc"]) / max(1, len(metrics["train_acc"]))
        logger.add_scalar("train_accuracy", float(avg_train_acc), global_step - 1)


        # example validation loop
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)

            # save val accuracies to be averaged (mean over the 10 samples)
            metrics["val_acc"].append(float(dummy_validation_accuracy.mean().item()))

        # log average val_accuracy at the end of the epoch
        avg_val_acc = sum(metrics["val_acc"]) / max(1, len(metrics["val_acc"]))
        logger.add_scalar("val_accuracy", float(avg_val_acc), global_step - 1)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)

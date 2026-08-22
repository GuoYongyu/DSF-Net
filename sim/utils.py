# coding: utf-8
from datetime import datetime
from typing import Iterable

# from matplotlib import pyplot as plt

from logger.logger import LOG


def save_training_process(
        dir: str,
        segment: str,
        epochs: int,
        crowding_weights: str,
        model_name: str,
        optimizer: str,
        scheduler: str,
        criterion: str,
        loss_weighting: str,
        task_weights: Iterable[float] | None,
        train_time: float,
        train_loss: dict,
        valid_loss: dict[str, dict],
        test_loss: dict[str, float],
    ):
    # save data
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    data_file = f"{dir}/{date_str}_{crowding_weights}_procedure.yaml"
    with open(data_file, "w") as f:
        f.write(f"total-train-time: {train_time:.4f}s\n")
        f.write(f"\nsegment: {segment}\n")
        f.write(f"\nnum-epochs: {epochs}\n")
        f.write(f"\nmodel-name: {model_name}\n")
        f.write(f"\noptimizer: {optimizer}\n")
        f.write(f"\nscheduler: {scheduler}\n")
        f.write(f"\ncriterion: {criterion}\n")
        f.write(f"\nloss-weighting: {loss_weighting}\n")
        f.write(f"\ntask-weights: {task_weights}\n")

        f.write(f"\nfinal-train-loss(Huber): {train_loss[epochs]}\n")

        f.write(f"\ntrain-loss(Huber)-history:\n")
        for epoch, loss in train_loss.items():
            f.write(f"  - epoch-{epoch}: {loss}\n")

        for name, loss in valid_loss.items():
            f.write(f"\nvalid-loss({name})-history:\n")
            for epoch, acc in loss.items():
                f.write(f"  - epoch-{epoch}: {acc}\n")

        f.write(f"\ntest-loss:\n")
        for name, loss in test_loss.items():
            f.write(f"  - {name}: {loss}\n")

    LOG.info(f"process of traning has been saved to {data_file}")

"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated

import numpy as np
from typer import Typer, Option

from automl.model import Models
from automl.dataset import Dataset
from automl.automl import AutoML
from automl.trainer import (
    Optimizer,
    LR_Scheduler,
    LossFn,
)
from automl.logging import set_log_level, LogLevel


app = Typer()


@app.command(
    help="Run the AutoML pipeline on a dataset.",
)
def auto(
    dataset: Annotated[
        Dataset,
        Option(
            help="The dataset to run on.",
        ),
    ],
    budget: Annotated[
        int,
        Option(
            help="The budget of the pipeline in seconds.",
        ),
    ] = 300,
    seed: Annotated[
        int,
        Option(
            help="Random seed for reproducibility.",
        ),
    ] = 42,
    quiet: Annotated[
        bool,
        Option(
            help="Whether to log only warnings and errors.",
        ),
    ] = False,
):
    if not quiet:
        set_log_level(LogLevel.INFO)
    else:
        set_log_level(LogLevel.WARNING)

    # Create AutoML pipeline
    automl = AutoML(
        dataset.factory,
        seed=seed,
    )

    # Run the pipeline
    automl.fit(budget=budget)

    # Predict on the test set, if it doesn't have labels (skin_cancer dataset)
    if dataset == "skin_cancer":
        test_preds = automl.predict()
        test_output_path = Path("data/exam_dataset/predictions.npy")
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # Else, evaluate the model on the test set
    else:
        print("Evaluating model on test set.")
        loss, accuracy = automl.evaluate()
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")


@app.command(
    help="Train a model on a dataset.",
)
def train(
    dataset: Annotated[
        Dataset,
        Option(
            help="The dataset to train on.",
        ),
    ],
    model: Annotated[
        Models,
        Option(
            help="The model to train.",
        ),
    ] = Models.ResNet50.value,
    epochs: Annotated[
        int,
        Option(
            help="The number of epochs to train for.",
        ),
    ] = 3,
    batch_size: Annotated[
        int,
        Option(
            help="The batch size.",
        ),
    ] = 64,
    optimizer: Annotated[
        Optimizer,
        Option(
            help="The optimizer to use.",
        ),
    ] = Optimizer.adamw.value,
    learning_rate: Annotated[
        float,
        Option(
            help="The learning rate.",
        ),
    ] = 1e-3,
    weight_decay: Annotated[
        float,
        Option(
            help="The weight decay.",
        ),
    ] = 1e-2,
    lr_scheduler: Annotated[
        LR_Scheduler,
        Option(
            help="The learning rate scheduler.",
        ),
    ] = LR_Scheduler.step.value,
    scheduler_step_size: Annotated[
        int,
        Option(
            help="The scheduler step size.",
        ),
    ] = 1000,
    scheduler_step_every_epoch: Annotated[
        bool,
        Option(
            help="Whether to step the scheduler every epoch.",
        ),
    ] = False,
    scheduler_gamma: Annotated[
        float,
        Option(
            help="The scheduler gamma.",
        ),
    ] = 0.1,
    loss_fn: Annotated[
        LossFn,
        Option(
            help="The loss function.",
        ),
    ] = LossFn.cross_entropy.value,
    seed: Annotated[
        int,
        Option(
            help="Random seed for reproducibility.",
        ),
    ] = 42,
    quiet: Annotated[
        bool,
        Option(
            help="Whether to log only warnings and errors.",
        ),
    ] = False,
):
    if not quiet:
        set_log_level(LogLevel.INFO)
    else:
        set_log_level(LogLevel.WARNING)

    print(f"Training on dataset {dataset.name}")

    automl = AutoML(
        dataset.factory,
        seed=seed,
    )

    pipeline_directory = Path(f"./results/{dataset.factory.__name__}_Train")

    if not pipeline_directory.exists():
        pipeline_directory.mkdir(parents=True)

    results = automl.run_pipeline(
        pipeline_directory=pipeline_directory,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        schedular_step_every_epoch=scheduler_step_every_epoch,
        loss_fn=loss_fn,
        results_file=pipeline_directory / "results.csv",
    )

    automl.trainer.save_model(pipeline_directory / "model.pth")

    print(f"Results: {results}")


if __name__ == "__main__":
    app()

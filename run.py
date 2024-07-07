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

from automl.data import Dataset
from automl.automl import AutoML
from automl.utils import set_log_level, Level, log as print


app = Typer()


@app.command()
def auto(
    dataset: Annotated[
        Dataset,
        Option(
            help="The dataset to run on.",
        ),
    ],
    output_path: Annotated[
        Path,
        Option(
            help="The path to save the predictions to.",
        ),
    ] = Path("predictions.npy"),
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
        set_log_level(Level.INFO)
    else:
        set_log_level(Level.WARNING)

    print(f"Fitting dataset {dataset.name}")

    automl = AutoML(
        dataset.factory,
        seed=seed,
    )
    automl.fit()
    # _, accuracy, test_preds = automl.predict()

    # # Write the predictions of X_test to disk
    # # This will be used by github classrooms to get a performance
    # # on the test set.
    # print("Writing predictions to disk")
    # with output_path.open("wb") as f:
    #     np.save(f, test_preds)

    # In case of running on the test data, also add the predictions.npy
    # to the correct location for autoevaluation.
    if dataset == "skin_cancer":
        test_output_path = Path("data/exam_dataset/predictions.npy")
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # check if test_labels has missing data
    if True:
        print(f"Accuracy on test set: {accuracy:.4f}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        print(f"No test split for dataset '{dataset}'")
    # # check if test_labels has missing data
    # if True:
    #     print(f"Accuracy on test set: {accuracy:.4f}")
    # else:
    #     # This is the setting for the exam dataset, you will not have access to the labels
    #     print(f"No test split for dataset '{dataset}'")


if __name__ == "__main__":
    app()

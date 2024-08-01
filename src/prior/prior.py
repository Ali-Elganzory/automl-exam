from enum import Enum

import yaml
import typer
from typing import Annotated, Optional
from pathlib import Path

app = typer.Typer()


class Confidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Optimizer(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"


def update_yaml_file(
        yaml_file: Path,
        batch_size: int = None,
        batch_size_confidence: str = None,
        optimizer: str = None,
        optimizer_confidence: str = None,
        learning_rate: float = None,
        learning_rate_confidence: str = None,
        weight_decay: float = None,
        weight_decay_confidence: str = None,
        scheduler_step_size: int = None,
        scheduler_step_size_confidence: str = None,
        scheduler_gamma: float = None,
        scheduler_gamma_confidence: str = None,
):
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file) or {}

    def update_parameter(param_name, default_value, default_confidence):
        if default_value is not None:
            yaml_data.setdefault(param_name, {})['default'] = default_value
        if default_confidence is not None:
            yaml_data.setdefault(param_name, {})['default_confidence'] = default_confidence.value

    update_parameter('batch_size', batch_size, batch_size_confidence)
    if optimizer:
        yaml_data.setdefault('optimizer', {}).setdefault('choices', [opt.value for opt in Optimizer])
        update_parameter('optimizer', optimizer.value if optimizer else None, optimizer_confidence)
        update_parameter('learning_rate', learning_rate, learning_rate_confidence)
        update_parameter('weight_decay', weight_decay, weight_decay_confidence)
        update_parameter('scheduler_step_size', scheduler_step_size, scheduler_step_size_confidence)
        update_parameter('scheduler_gamma', scheduler_gamma, scheduler_gamma_confidence)
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(yaml_data, file, default_flow_style=False, sort_keys=False)


@app.command()
def update_config(
        batch_size: Annotated[
            int,
            typer.Option(help="Batch size for training."),
        ] = 64,
        batch_size_confidence: Annotated[
            Optional[Confidence],
            typer.Option(help="Confidence level of the default batch size value (e.g., low, medium, high)."),
        ] = Confidence.MEDIUM.value,
        optimizer: Annotated[
            Optional[Optimizer],
            typer.Option(help="Optimizer to use."),
        ] = Optimizer.ADAMW.value,
        optimizer_confidence: Annotated[
            Optional[Confidence],
            typer.Option(help="Confidence level of the default optimizer value (e.g., low, medium, high)."),
        ] = Confidence.MEDIUM.value,
        learning_rate: Annotated[
            float,
            typer.Option(help="Learning rate for the optimizer."),
        ] = 0.001,
        learning_rate_confidence: Annotated[
            Optional[Confidence],
            typer.Option(help="Confidence level of the default learning rate value (e.g., low, medium, high)."),
        ] = Confidence.MEDIUM.value,
        weight_decay: Annotated[
            float,
            typer.Option(help="Weight decay for regularization."),
        ] = 0.01,
        weight_decay_confidence: Annotated[
            Optional[Confidence],
            typer.Option(help="Confidence level of the default weight decay value (e.g., low, medium, high)."),
        ] = Confidence.LOW.value,
        scheduler_step_size: Annotated[
            int,
            typer.Option(help="Scheduler step size for learning rate adjustment."),
        ] = 1000,
        scheduler_step_size_confidence: Annotated[
            Optional[Confidence],
            typer.Option(help="Confidence level of the default scheduler step size value (e.g., low, medium, high)."),
        ] = Confidence.MEDIUM.value,
        scheduler_gamma: Annotated[
            float,
            typer.Option(help="Scheduler gamma for learning rate adjustment."),
        ] = 0.1,
        scheduler_gamma_confidence: Annotated[
            Optional[Confidence],
            typer.Option(help="Confidence level of the default scheduler gamma value (e.g., low, medium, high)."),
        ] = Confidence.MEDIUM.value,
):
    update_yaml_file(
        yaml_file=Path("test.yaml"),
        batch_size=batch_size,
        batch_size_confidence=batch_size_confidence,
        optimizer=optimizer,
        optimizer_confidence=optimizer_confidence,
        learning_rate=learning_rate,
        learning_rate_confidence=learning_rate_confidence,
        weight_decay=weight_decay,
        weight_decay_confidence=weight_decay_confidence,
        scheduler_step_size=scheduler_step_size,
        scheduler_step_size_confidence=scheduler_step_size_confidence,
        scheduler_gamma=scheduler_gamma,
        scheduler_gamma_confidence=scheduler_gamma_confidence,
    )


if __name__ == "__main__":
    app()

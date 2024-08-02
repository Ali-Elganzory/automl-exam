"""Utility functions for reading and processing results."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

import neps
import pandas as pd
from neps.utils.types import _ConfigResultForStats


def process_seed(
        *,
        path: str | Path,
        seed: str | int | None,
        key_to_extract: str | None = None,
        consider_continuations: bool = False,
        n_workers: int = 1,
) -> tuple[list[float], list[float], float]:
    """Reads and processes data per seed."""
    path = Path(path)
    if seed is not None:
        path = path / str(seed) / "summary_csv" / "config_data.csv"

    df = pd.read_csv(path)
    records = df.to_dict(orient='records')

    def nest_result_fields(record):
        nested_record = {}
        nested_record['config_id'] = record['config_id']
        nested_record['status'] = record['status']
        nested_record['config'] = {
            'batch_size': record['config.batch_size'],
            'epochs': record['config.epochs'],
            'learning_rate': record['config.learning_rate'],
            'optimizer': record['config.optimizer'],
            'scheduler_gamma': record['config.scheduler_gamma'],
            'scheduler_step_size': record['config.scheduler_step_size'],
            'weight_decay': record['config.weight_decay']
        }
        nested_record['metadata'] = {
            'account_for_cost': record['metadata.account_for_cost'],
            'eval_cost': record['metadata.eval_cost'],
            'max': record['metadata.max'],
            'time_end': record['metadata.time_end'],
            'time_sampled': record['metadata.time_sampled'],
            'used': record['metadata.used']
        }
        nested_record['result'] = {
            'cost': record['result.cost'],
            'loss': record['result.loss'],
            'info_dict': {
                'cost': record['result.info_dict.cost'],
                'train_accuracies': record['result.info_dict.train_accuracies'],
                'train_losses': record['result.info_dict.train_losses'],
                'train_time': record['result.info_dict.train_time'],
                'val_accuracies': record['result.info_dict.val_accuracies'],
                'val_losses': record['result.info_dict.val_losses']
            }
        }
        return _ConfigResultForStats(id=record['config_id'], config=nested_record['config'],
                                     result=nested_record['result'], metadata=nested_record['metadata'])

    nested_records = {record['config_id']: nest_result_fields(record) for record in records}
    stats = nested_records
    sorted_stats = sorted(sorted(stats.items()), key=lambda x: len(x[0]))
    stats = OrderedDict(sorted_stats)

    # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
    max_cost: float = -1.0
    if key_to_extract == "fidelity":
        # TODO(eddiebergman): This can crash for a number of reasons, namely if the config
        # crased and it's result is an error, or if the `"info_dict"` and/or
        # `key_to_extract` doesn't exist
        max_cost = max(s.result["info_dict"][key_to_extract] for s in stats.values())  # type: ignore

    global_start = stats[min(stats.keys())].metadata["time_sampled"]

    def get_cost(idx: str) -> float:
        if key_to_extract is not None:
            # TODO(eddiebergman): This can crash for a number of reasons, namely if the
            # config crased and it's result is an error, or if the `"info_dict"` and/or
            # `key_to_extract` doesn't exist
            return float(stats[idx].result["info_dict"][key_to_extract])  # type: ignore

        return 1.0

    losses = []
    costs = []

    for config_id, config_result in stats.items():
        config_cost = get_cost(config_id)
        if consider_continuations:
            if n_workers == 1:
                # calculates continuation costs for MF algorithms NOTE: assumes that
                # all recorded evaluations are black-box evaluations where
                # continuations or freeze-thaw was not accounted for during optimization
                if "previous_config_id" in config_result.metadata:
                    previous_config_id = config_result.metadata["previous_config_id"]
                    config_cost -= get_cost(previous_config_id)
            else:
                config_cost = config_result.metadata["time_end"] - global_start

        # TODO(eddiebergman): Assumes it never crashed and there's a loss available,
        # not fixing now but it should be addressed
        losses.append(config_result.result["loss"])  # type: ignore
        costs.append(config_cost)

    return list(np.minimum.accumulate(losses)), costs, max_cost

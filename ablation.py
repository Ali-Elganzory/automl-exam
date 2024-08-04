import re
import yaml
import ast

from automl.automl import AutoML
from automl.dataset import Datasets
from automl.trainer import LR_Scheduler, LossFn

import csv


def greedy_ablation(start_config, end_config):
    """
    Performs greedy ablation

    Args:
        start_config: The starting configuration of the algorithm.
        end_config: The ending configuration of the algorithm.

    Returns:
        A list of ablation steps, where each step is a change to the configuration.
    """
    ablation_path = []
    modified_indices = []
    best_local_config = start_config.copy()
    losses = []

    for t in range(6):
        best_local_config, loss, modified_indice = best_modified_config(best_local_config, end_config, modified_indices)
        if modified_indice is not None:
            modified_indices.append(modified_indice)
        else:
            break

        ablation_path.append(best_local_config.copy())
        losses.append(loss)

    with open("src/greedy_ablation/ablation_results.csv", 'w', newline='') as csvfile:
        fieldnames = ['Step', 'Configuration', 'Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, (config, loss) in enumerate(zip(ablation_path, losses)):
            writer.writerow({'Step': i + 1, 'Configuration': config, 'Loss': loss})

    print(f"ablation path: {ablation_path}")

    return ablation_path


def best_modified_config(start_config, best_config, modified_indices):
    best_loss = float('inf')
    best_config_result = None
    best_key = None

    for key in best_config.keys():
        if key not in modified_indices:
            modified_config = start_config.copy()

            modified_config[key] = best_config[key]
            print(f"Start config: {start_config}")
            print(f"modified config:{modified_config}")
            print(f"best config:{best_config}")

            automl = AutoML(
                Datasets.EMOTIONS.factory,
                seed=42,
            )

            results = automl.run_pipeline(
                epochs=1,  # TODO change this to 30
                batch_size=modified_config['batch_size'],
                optimizer=modified_config['optimizer'],
                learning_rate=modified_config['learning_rate'],
                weight_decay=modified_config['weight_decay'],
                lr_scheduler=LR_Scheduler.step,
                scheduler_step_size=modified_config['scheduler_step_size'],
                scheduler_gamma=modified_config['scheduler_gamma'],
                schedular_step_every_epoch=False,
                loss_fn=LossFn.cross_entropy
            )

            current_loss = results['loss']
            if current_loss < best_loss:
                best_loss = current_loss
                best_config_result = modified_config
                best_key = key

    return best_config_result, best_loss, best_key


def read_start_config(config_space):
    config = {}
    for key, value in config_space.items():
        if 'default' in value and key != 'epochs':
            config[key] = value['default']
        else:
            print(f"No default value specified for {key}")

    return config


def reorder_dict_keys(reference_dict, target_dict):
    """
    Reorders the target_dict to match the key order of reference_dict.

    Args:
        reference_dict (dict): The dictionary whose key order should be used.
        target_dict (dict): The dictionary to be reordered.

    Returns:
        dict: A new dictionary with keys ordered to match reference_dict.
    """
    key_order = list(reference_dict.keys())

    reordered_dict = {key: target_dict[key] for key in key_order if key in target_dict}
    return reordered_dict


if __name__ == "__main__":
    with open('pipeline_space.yaml', 'r') as file:
        config_space = yaml.safe_load(file)

    start_config = read_start_config(config_space)

    print(f"Start Config: {start_config}")

    # TODO ablation for skin cancer dataset on 30 epochs
    with open('results/benchmark=Flowers/algorithm=PriorBand-BO/seed=42/best_loss_with_config_trajectory.txt',
              'r') as file:
        data = file.read()

    pattern = r"Loss: ([\d\.]+)\nConfig ID: ([\d_]+)\nConfig: ({.*?})"
    matches = re.findall(pattern, data, re.DOTALL)

    best_config = ast.literal_eval(matches[-1][2])
    best_config = reorder_dict_keys(start_config, best_config)
    print(f"Best Config: {best_config}")

    greedy_ablation(start_config, best_config)

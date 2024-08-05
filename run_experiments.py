import os
from time import sleep
from concurrent.futures import ThreadPoolExecutor

from automl.dataset import Datasets


def command_factory(
    dataset: Datasets,
    seed: int,
):
    return f"python run.py auto --dataset {dataset.value} --seed {seed} --budget 82800"


commands = [
    command_factory(
        Datasets.FASHION,
        71,
    ),
    command_factory(
        Datasets.FLOWERS,
        71,
    ),
    command_factory(
        Datasets.EMOTIONS,
        71,
    ),
    command_factory(
        Datasets.SKIN_CANCER,
        71,
    ),
]

gpu_available = [True]


def run_command(command: str, gpu_id: int):
    command = f"{command}"
    os.system(command)
    gpu_available[gpu_id] = True


def run_experiments(commands):
    with ThreadPoolExecutor(max_workers=4) as executor:
        for command in commands:
            while not any(gpu_available):
                sleep(1)

            gpu_id = gpu_available.index(True)
            gpu_available[gpu_id] = False
            executor.submit(run_command, command, gpu_id)


if __name__ == "__main__":
    run_experiments(commands)

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from plots.incumbents_plots import Benchmark


def gather_csv_files(base_path, benchmarks):
    """Collect all CSV file paths for the given benchmarks"""
    csv_files = []
    for benchmark in benchmarks:
        benchmark_path = Path(base_path) / f"benchmark={benchmark}/algorithm=PriorBand-BO"
        if benchmark_path.exists():
            seeds = sorted(os.listdir(benchmark_path))
            for seed in seeds:
                seed_path = benchmark_path / seed
                csv_file = seed_path / 'best_config_results.csv'
                if csv_file.exists():
                    csv_files.append(csv_file)
    return csv_files


def load_data(csv_files):
    """Load data from the list of CSV files into a dictionary"""
    benchmark_dfs = {}
    for file in csv_files:
        parts = file.parts
        benchmark_name = parts[-4].split('=')[1]
        seed = parts[-2].split('=')[1]
        df = pd.read_csv(file)
        benchmark_dfs[(benchmark_name, seed)] = df
    return benchmark_dfs


def plot_data(benchmark_dfs, baselines, output_path):
    """Plot training and validation loss and accuracy for each benchmark and seed"""
    num_benchmarks = len(benchmark_dfs)
    num_rows = (num_benchmarks * 2 + 1) // 2

    plt.figure(figsize=(18, num_rows * 6))

    for i, ((name, seed), df) in enumerate(benchmark_dfs.items()):
        # Plot Loss
        plt.subplot(num_rows, 2, i * 2 + 1)
        plt.plot(df['epoch'], df['train_loss'], color='y', label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], color='b', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} (Seed: {seed}) - Loss')
        plt.legend(loc='best')
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(num_rows, 2, i * 2 + 2)
        plt.plot(df['epoch'], df['train_accuracy'], color='y', label='Train Accuracy')
        plt.plot(df['epoch'], df['val_accuracy'], color='b', label='Val Accuracy')
        plt.axhline(y=baselines[name], color='r', linestyle='--', label='Baseline Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{name} (Seed: {seed}) - Accuracy')
        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    base_path = '../../results/'
    output_path = Path(base_path) / "incumbent_train_acc_plot.png"
    # TODO make probabilistic once we have other runs
    benchmarks = [Benchmark.FASHION.value, Benchmark.FLOWERS.value, Benchmark.EMOTIONS.value]
    baselines = {
        Benchmark.FASHION.value: 0.88,
        Benchmark.EMOTIONS.value: 0.4,
        Benchmark.FLOWERS.value: 0.55,
        Benchmark.SKIN_CANCER.value: 0.71
    }

    csv_files = gather_csv_files(base_path, benchmarks)
    benchmark_dfs = load_data(csv_files)
    plot_data(benchmark_dfs, baselines, output_path)

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # TODO change csv file paths
    csv_files = ['../../results/Fashion_results.csv', '../../results/Flowers_results.csv']

    benchmark_dfs = {}

    for file in csv_files:
        benchmark_name = file.split('/')[-1].replace('_results.csv', '')
        df = pd.read_csv(file)
        benchmark_dfs[benchmark_name] = df

    num_benchmarks = len(benchmark_dfs)
    num_rows = (num_benchmarks * 2 + 1) // 2

    plt.figure(figsize=(18, num_rows * 6))

    for i, (name, df) in enumerate(benchmark_dfs.items()):

        plt.subplot(num_rows, 2, i * 2 + 1)
        plt.plot(df['epoch'], df['train_loss'], color='b', linestyle='-', label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], color='r', linestyle='--', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} - Loss')
        plt.legend(loc='best')
        plt.grid(True)  # Add grid

        plt.subplot(num_rows, 2, i * 2 + 2)
        plt.plot(df['epoch'], df['train_accuracy'], color='b', linestyle='-', label='Train Accuracy')
        plt.plot(df['epoch'], df['val_accuracy'], color='r', linestyle='--', label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{name} - Accuracy')
        plt.legend(loc='best')
        plt.grid(True)  # Add grid

    plt.tight_layout()
    plt.savefig("train_acc_plot.png")

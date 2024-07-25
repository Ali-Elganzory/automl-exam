from enum import Enum
from typer import Typer, Option
from plots.plot import plot
from typing import List, Annotated, Optional

app = Typer()


class Benchmark(Enum):
    FASHION = "Fashion"
    FLOWERS = "Flowers"
    EMOTIONS = "Emotions"
    SKIN_CANCER = "Skin_cancer"


@app.command(help="Generate plots for finished runs")
def plots(
        benchmarks: Annotated[
            Optional[List[Benchmark]],
            Option(
                help=(
                        "Specify the dataset(s) to generate plots for. "
                        "You can provide multiple benchmarks by using this option multiple times. "
                        "For example: --benchmark FASHION --benchmark EMOTIONS. "
                        "If not provided, plots will be generated for all benchmarks."
                ),
            ),
        ] = None,
):
    if benchmarks is None:
        benchmarks = [benchmark for benchmark in Benchmark]

    root_directory = "./results"
    scientific_mode = True
    algorithms = ["PriorBand-BO"]

    plot(
        root_directory=root_directory,
        scientific_mode=scientific_mode,
        benchmarks=[benchmark.value for benchmark in benchmarks],
        algorithms=algorithms,
    )


if __name__ == "__main__":
    app()

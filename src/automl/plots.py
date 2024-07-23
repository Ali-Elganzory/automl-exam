import neps

if __name__ == "__main__":
    root_directory = "../../results"

    neps.plot(
        root_directory=root_directory,
        scientific_mode=True,
        benchmarks=["Flowers"],
        algorithms=["PriorBand-BO"]
    )

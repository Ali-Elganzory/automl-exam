# Installing the project

```bash
pip install -e .
```

# Running the AutoML pipeline

To get help on the arguments, run the following command:
```bash
python run.py --help
```

To run the AutoML pipeline, run the following command:
```bash
python run.py auto --dataset <dataset_name> --seed <seed> --budget <budget in seconds>
```
This will print out the best configuration found by the pipeline, as well as saving the best model to `results/benchmark=<dataset_name>/algorithm=PriorBand-BO/seed=<seed>/best_config_model.pth`.

## AutoML on Skin Cancer dataset (23 hours budget)
```bash
python run.py auto --dataset skin_cancer --seed 71 --budget 82800
```

# Generating Predictions

To generate the predictions of the best model found by the AutoML pipeline, run the following command:
```bash
python run.py predict --dataset skin_cancer --model-path "results/benchmark=SkinCancer/algorithm=PriorBand-BO/seed=71/best_config_model.pth" --predictions-path "./data/exam_dataset/predictions.npy"
```
The predictions will be saved in the project root as `
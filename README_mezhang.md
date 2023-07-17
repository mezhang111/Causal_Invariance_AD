# Causal Invariance in AD, features implemented by Michael

## To Start

Navigate to submodules/anomalib, pip install -e . \
Navigate to submodules/diagvib-6, pip install -e . \

## TODOS
Use Hydra for configs \


## Mead Env Generation

Put mnist.npz and mnist_processed.npz in tmp folder. Then do "python generate_mead_env.py --root path_to_env_yml_configs" \
The dataset would be generated under path_to_env_yml_configs/dataset \

## Quickstart: Train And Test

Create your own config file in /configs or use the existing ones. \
Then do: python train.py --model path_to_model_config --dataset path_to_dataset_config\
Same for test: python train.py --model path_to_model_config --dataset path_to_dataset_config --weight path_to_weight_file


## Config Examples:

All the configs can be found in configs/. \
Right now it is split in: datasets, preprocess and models, each contains a part of the final config for training and testing. \
The evaluation config is in configs/eval

## New Dataset for mead: FolderDg
Two classes, `FolderDgDataset` and `FolderDg`, for handling datasets in a folder structure. \
The FolderDg shall be used for lightning module. \
The code can be broken down into the following sections:

1. **Importing libraries and defining the `__all__` variable.**

2. **`FolderDgDataset` class:**
   - Inherits from `FolderDataset`.
   - `__init__(self, **kwargs)`: Initializes the object with additional `train_env_labels` attribute.
   - `__getitem__(self, index: int)`: Retrieves an item from the dataset and includes the environment label if available.
   - `set_train_env_labels(self)`: Sets the environment labels for the training dataset by reading them from a CSV file.
   - `from_folder_dataset(cls, folder: FolderDataset) -> FolderDgDataset`: Creates a new `FolderDgDataset` instance from a given `FolderDataset` instance.
   - `get_init_args(cls) -> list`: Returns a list of arguments required to initialize the parent class `FolderDataset`.

3. **`FolderDg` class:**
   - Inherits from `Folder`.
   - `__init__(self, **kwargs)`: Initializes the object by converting the `train_data` and `test_data` attributes to `FolderDgDataset` instances.
   - `setup(self, stage: str | None = None) -> None`: Sets up the dataset by generating environment and calling the parent class `setup()` method. Also, sets the environment labels for the training dataset.
   - `get_init_args(cls) -> set`: Returns a set of arguments required to initialize the parent class `Folder`.

The script allows handling datasets with a folder structure and enables the use of environment labels in the data.

## Wilds For Mead
1. Define a function `album_transform_for_wilds` to wrap an Albumentations transform for compatibility with WILDS datasets.

2. Define a class `WildsAnomaly` which inherits from WILDSSubset, providing functionality for selecting samples with specific labels (normal or anomalous) from a WILDSDataset.

3. Define a class `WildsAnomalyLightning` which inherits from LightningDataModule, providing functionality for creating train, validation, and test data loaders with the specified transformations and label selections.
   - The constructor takes in various arguments such as batch sizes, number of workers, WILDSDataset, normal and anomaly labels, a grouper object, and a transform configuration.
   - Implement methods for creating data loaders for training, validation, and testing phases.

To use this DataModule, you would create an instance of the `WildsAnomalyLightning` class with the desired configurations, and pass it to a PyTorch Lightning Trainer for training and evaluation.



## Training script explanation
1. **Importing libraries and initializing logger.**

2. **Utility functions:**
   - `_snake_to_pascal_case(model_name: str)`: Converts a model name from snake_case to PascalCase.
   - `seed_everything_at_once(seed=42)`: Sets random seeds for various libraries to ensure reproducibility.
   - `get_args() -> Namespace`: Parses command line arguments required for training.
   - `preprocess_config(config)`: Preprocesses the provided configuration.

3. **`train()`: Main function to train the anomaly detection model.**
   - Loads the configuration files for the model and dataset.
   - Preprocesses the configuration.
   - Sets random seeds for reproducibility (if provided).
   - Loads the data module based on the dataset format specified in the configuration.
   - Loads the model.
   - Initializes the logger and callbacks for the PyTorch Lightning Trainer.
   - Trains the model using the Trainer.
   - Loads the best model weights based on validation performance.
   - Tests the model if a test set is provided.

4. **`load_model(config)`: Loads the model based on the provided configuration.**

5. **`load_data_module(config, dataset_format)`: Loads the data module based on the dataset format specified in the configuration.**

6. **Main script entry point:**
   - Calls the `train()` function to start the training process.

To use this script, you need to provide configuration files for the model and dataset using the command line arguments `--model` and `--dataset`. Optionally, you can set the logging level using the `--log-level` argument.

## tools/generate_ood_config.py
Given a base env config, the script generates configs for all possible ood envs for each feature. \
For instance, when yellow is used in base env, the code would generate ood_envs for other hues except yellow \

The code does the following:

1. Define global variables for features, textures, hues, scales, lightness, and positions.

2. Define a function `get_args` to parse command-line arguments.

3. Define a function `get_used_features` to extract the used features from the base environment configuration.

4. Define the main function `generate`:
   - Parse command-line arguments.
   - Load the base environment configuration.
   - Get used features from the base configuration.
   - Determine the features to be used for generating OD environments.
   - For each OOD feature and its corresponding values:
     - Create a new output directory for the OOD feature.
     - For each OOD value:
       - Create a new OOD environment configuration based on the template.
       - Update the configuration with the used features and the current OOD value.
       - Save the configuration as a YAML file in the output directory.

5. Run the main function `generate` when the script is executed.

To use this script, provide the required command-line arguments for the base environment configuration file and the output directory. The script will generate new OD environment configurations and save them as YAML files in the specified output directory.

## evaluation.py

This script evaluates an anomaly detection model with an evaluation config (configs/evaluation/eval.yaml). \
An example evaluation configuration file:

- Specify the paths to the test configuration files and corresponding checkpoint files.
- Set the project path to save the evaluation results.
- One config corresponds to one ckpt
- Configure visualization settings:
  - show_images: Whether to display images on the screen.
  - save_images: Whether to save images to the filesystem.
  - log_images: Whether to log images to the available loggers (if any).
- List the datasets to be used for evaluation. For each dataset:
  - Specify the dataset name, path, normal training data directory, abnormal test data directory, and normal test data directory.
- Each ckpt would be evaluated on all the datasets


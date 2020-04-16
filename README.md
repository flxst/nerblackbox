# nlp-swebert-applications 

This repository provides support for fine-tuning a pretrained transformer model
on any dataset for the following `Downstream Tasks`:
- Named Entity Recognition (NER)

## Installation
1. Clone this repository: `git clone <url>`
2. Install package: `pip install .`
3. Install GPU support (recommended): `bash setup_gpu_support.sh`


## Usage

Fine-tune a model in a few simple steps, 
using either 
- the command line interface (CLI) or 
- the Python API.

1.  Initialization:

    ```
    # CLI
    nerbb --init  
    
    # Python API
    nerbb.init()
    ```
    
    This creates a `./data` directory with the following structure:
    ```
    data/
    └── datasets
        └── conll2003           # built-in dataset (english)
            └── train.csv
            └── val.csv
            └── test.csv
        └── swedish_ner_corpus  # built-in dataset (swedish)
            └── train.csv
            └── val.csv
            └── test.csv
    └── experiment_configs
        └── experiment_default.ini
    └── pretrained_models       # custom model checkpoints
    └── results
    ```

2. run experiment

    ```
    # e.g. <experiment_name> = 'experiment_default'
    ```
    ```
    # CLI
    nerbb --run-experiment <experiment_name>  
    
    # Python API
    nerbb.run_experiment(<experiment_name>)
    ```
    metrics & results are logged in directory `./data/results` with 
    - mlflow, display with `mlflow ui`
    - tensorboard, display with `tensorboard --logdir tensorboard --reload_multifile=true`
    
3. inspect results:

        # CLI
        nerbb --get-experiment-results <experiment_name>
        
        # Python API
        nerbb.get_experiment_results(<experiment_name>)                 
        

This works out of the box for built-in datasets and models.
Custom datasets and models can also be included.

## Datasets and Models
    
- Built-in datasets:
    - CoNLL2003 (english)
    - Swedish NER Corpus (swedish)
   
- Built-in models:
    - All built-in or community-uploaded BERT models of the transformers library 
        
### Custom Datasets
 
To include your own custom dataset, do the following:
- Create a new module `datasets/formatter/<new_dataset>_formatter.py`
- Derive the class `<NewDataset>Formatter` from `BaseFormatter` and implement the abstract base methods

### Custom Models
 
To include a new model, do the following:
 - Create a new folder `pretrained_model/<new_model>`. The folder name must include the architecture type, e.g. `bert`
 - Add the following files to the folder:
    - `config.json`
    - `pytorch_model.bin`
    - `vocab.txt`
    

# nlp-swebert-applications 

This repository provides support for fine-tuning a pretrained transformer model
on any dataset for the following `Downstream Tasks`:
- Named Entity Recognition (NER)

## Installation
1. Clone this repository: `git clone <url>`
2. Install package: `pip install -e .`
3. Install GPU support (recommended): `bash setup_gpu_support.sh`


## Usage

Fine-tune a model in a few simple steps, 
using either 
- the CLI (command line interface) or 
- the Python API.
```
# CLI
nerbb --help

# Python API
from ner_black_box import NerBlackBox
nerbb = NerBlackBox()
```

#### 1.  Initialization:

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
    └── default.ini
    └── exp0.ini
└── pretrained_models       # custom model checkpoints
└── results
```

#### 2. Single Experiment

- A single experiment is defined through a configuration 
file in `./data/experiment_configs/<experiment_name>.ini`

- The following parameters need to be specified in the configuration file:
    - dataset
    - pretrained model
    - hyperparameters 
    
- Note that an experiment can entail multiple training runs with different 
hyperparameter combinations.

- See `./data/experiment_configs/exp0.ini` for an example

---
Once an experiment is defined, the following commands can be used to 
run the experiment and access its results:

a. Show Experiment Configuration

```
# CLI
nerbb --show_experiment_config <experiment_name>  

# Python API
nerbb.show_experiment_config(<experiment_name>)
```

b. Run Experiment

```
# CLI
nerbb --run_experiment <experiment_name>  

# Python API
nerbb.run_experiment(<experiment_name>)
```
   
c. Access detailed run histories & results using either mlflow or tensorboard (CLI):

- `nerbb --mlflow` [+ enter http://localhost:5000 in your browser]
- `nerbb --tensorboard`[+ enter http://localhost:6006 in your browser] 

d. Get main results:

```
# CLI
nerbb --get_experiment_results <experiment_name>  # prints overview on runs

# Python API
experiment_results = nerbb.get_experiment_results(<experiment_name>)

experiment_results.experiment        # data frame with overview on experiment
experiment_results.single_runs       # data frame with overview on single runs
experiment_results.average_runs      # data frame with overview on average runs
experiment_results.best_single_run   # dictionary with overview on best single run
experiment_results.best_average_run  # dictionary with overview on best average run
experiment_results.best_model        # pytorch model of best single run  
```
        
e. Use best model for predictions (only python API):

```
# e.g. <text_input> = ['some text that needs to be tagged']
```
```
# Python API
experiment_results.best_model.predict(<text_input>)
```
   
#### 3. All Experiments

a. Get Experiments Overview

```
# CLI
nerbb --get_experiments  

# Python API
nerbb.get_experiments()
```
   
b. Get Best Runs Overview:
```
# CLI
nerbb --get_experiments_results

# Python API
experiments_results = nerbb.get_experiments_results()
experiments_results.best_single_runs    # data frame with overview on best single runs
experiments_results.best_average_runs   # data frame with overview on best average runs
```
        
## Datasets and Models

The above works out of the box for built-in datasets and models.
    
- Built-in datasets:
    - CoNLL2003 (english)
    - Swedish NER Corpus (swedish)
   
- Built-in models:
    - All built-in or community-uploaded BERT models of the transformers library 
        
### Custom Datasets
 
To include your own custom dataset, do the following:
 - Create a new folder `./data/datasets/<new_dataset>`.
 - Add the following files to the folder:
    - `train.csv`
    - `val.csv`
    - `test.csv`
     - (todo: additional information on csv format needed here)
    
Own custom datasets can also be created programmatically (like the built-in datasets):
 - (todo: revise the following)
 - Create a new module `./data/datasets/formatter/<new_dataset>_formatter.py`
 - Derive the class `<NewDataset>Formatter` from `BaseFormatter` and implement the abstract base methods
 - (todo: additional instructions needed here)

### Custom Models
 
To include your own custom model, do the following:
 - Create a new folder `./data/pretrained_model/<new_model>`. The folder name must include the architecture type, e.g. `bert`
 - Add the following files to the folder:
    - `config.json`
    - `pytorch_model.bin`
    - `vocab.txt`

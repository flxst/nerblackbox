# nlp-swebert-applications 

Arbetsförmedlingen (The Swedish Public Employment Service) has developed Swedish 
BERT models (SweBERT) which are available [here](https://github.com/af-ai-center/SweBERT).

This repository provides support for the following 
`Downstream Task Applications`:
- Named Entity Recognition (NER)

Their use is demonstrated on publicly available datasets.

  
## Getting Started

It is recommended to run all commands, scripts and notebooks in a virtual environment.

### Setup

- Install packages:


    pip install -r requirements.txt
    bash setup_apex.sh                            # installs apex from https://github.com/NVIDIA/apex
    
    # on linux, reinstall torch torchvision (for cuda support):
    pip uninstall torch torchvision
    pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    
        
## Dataset setup

Built-in datasets are:
 - CoNLL2003 (english)
 - Swedish Umeå Corpus (swedish)
 - Swedish NER Corpus (swedish)
 
To include a new dataset, do the following:
- Create a new module `datasets/formatter/<new_dataset>_formatter.py`
- Derive the class `<NewDataset>Formatter` from `BaseFormatter` and implement the abstract base methods

Setup a dataset (i.e. create *.csv files in directory `./datasets/ner`):


    # e.g. <ner_dataset> = swedish_ner_corpus, suc, conll2003
    python scripts/script_setup_dataset.py --ner_dataset <ner_dataset>   
    

## Model setup
Built-in models are:
 - All built-in or community-uploaded models of the transformers library
 
To include a new model, do the following:
 - Create a new folder `<new_model>` in the project's root directory. The folder name must include the architecture type, e.g. `bert`
 - Add the following files to the folder:
    - `config.json`
    - `pytorch_model.bin`
    - `vocab.txt`
    
## Train a model on a dataset

- Experiment: 

    recommended, allows for single or multiple runs
    
    - specify dataset, model & hyperparameters in `experiment_configs/<experiment_name>.ini`
    - call
        
        `. run_experiment.sh <experiment_name>` to execute all runs

        `. run_experiment.sh <experiment_name> <run_name>` to execute single run
        

- Single Run: 

    alternative

    `python script_bert_ner_single.py` 

  - flags:
    - `--experiment_name * --run_name *` to structure runs 
    - `--pretrained_model_name * --dataset_name *` to choose model & dataset
    - `--max_epochs * --lr_max * [..]` to choose hyperparameters 

  
- metrics & results are logged in directory `results` with 
    - mlflow, display with `mlflow ui`
    - tensorboard, display with `tensorboard --logdir tensorboard --reload_multifile=true`


## Notebooks (prototypes)

- `bert_ner.ipynb` shows how to train SweBERT on the 
NER downstream task using the publicly available Swedish NER corpus dataset.  

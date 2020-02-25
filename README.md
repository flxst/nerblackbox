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
    
        
- Download and preprocess a `<ner_dataset>` to directory `./datasets/ner`:


    # <ner_dataset> = swedish_ner_corpus, SUC
    bash setup_datasets.sh <ner_dataset>   
    
## Scripts

- `bert_ner.py`: train SweBERT on NER downstream task. 

  - flags:
    - ###### `--experiment_name * --run_name *` to structure runs 
    - `--pretrained_model_name * --dataset_name *` to choose model & dataset
    - `--num_epochs * --lr_max * [..]` to choose hyperparameters 
  - metrics & results are logged in directory `results` with 
    - mlflow, display with `mlflow ui`
    - tensorboard, display with `tensorboard --logdir tensorboard`

  
- `mlflow run mlflow_experiments/test --no-conda`


## Notebooks (prototypes)

- `bert_ner.ipynb` shows how to train SweBERT on the 
NER downstream task using the publicly available Swedish NER corpus dataset.  

- `bert_ner_plot_metrics.ipynb` displays training and evaluation metrics.



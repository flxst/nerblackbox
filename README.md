# nlp-swebert-applications 

Arbetsförmedlingen (The Swedish Public Employment Service) has developed Swedish 
BERT models (SweBERT) which are available [here](https://github.com/af-ai-center/SweBERT).

This repository provides support for the following 
`Downstream Task Applications`:
- Named Entity Recognition (NER)

Their use is demonstrated on publicly available datasets.

  
## Getting Started

It is recommended to run all commands and notebooks within a virtual environment.

### Setup

- Install packages:


    pip install -r requirements.txt
    bash setup_apex.sh                            # installs apex from https://github.com/NVIDIA/apex
    
    # on linux, reinstall torch torchvision (for cuda support):
    pip uninstall torch torchvision
    pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    
        
- Download datasets to directory `./datasets`:


    bash setup_datasets.sh <ner_dataset>   # <ner_dataset> = swedish_ner_corpus, SUC
    
### Notebooks (to get started / prototypes)

- `bert_ner.ipynb` shows how to train SweBERT on the 
NER downstream task using the publicly available Swedish NER corpus dataset.  

- `bert_ner_plot_metrics.ipynb` displays training and evaluation metrics.


### Scripts

- `bert_ner.py`: train SweBERT on NER downstream task. 

  - metrics & results are logged with mlflow
  - use flags `--experiment_name <experiment_name> --run_name <run_name>` to structure runs 
  - display metrics & results with `mlflow ui`

  
- `mlflow run mlflow_experiments/test --no-conda`
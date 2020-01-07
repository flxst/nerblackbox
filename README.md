# nlp-bert

This repository provides 

- Arbetsförmedlingens Swedish BERT models
- Tools for downstream task application using publicly available datasets


## Overview

BERT and other transformer models are readily available in different
languages from the transformers package. 

Arbetsförmedlingen (Swedish Public Employment Service) has developed 
Swedish BERT models which were trained on Swedish Wikipedia with approximatelly 2 million articles and 300 million words.

## Models

The Swedish BERT models are provided in the format used by the transformers package. 

This means that both tokenizer and model can be instantiated using the `from_pretrained()` method 
of the BERT-related transformers classes like so:

    tokenizer = BertTokenizer.from_pretrained(<model type>)
    
    model = BertModel.from_pretrained(<model type>)
    model = BertForTokenClassification.from_pretrained(<model type>) 
    ..


The following model types are available:
- `swe-bert-base`

    12-layer, 768-hidden, 12-heads, 110M parameters
     
    (zipped: ~1.5 GB, unzipped: ~1.7 GB)
- `swe-bert-large`

    24-layer, 1024-hidden, 16-heads, 340M parameters
    
    (zipped: ~x.x GB, unzipped: ~x.x GB)
    
For download see the "Setup" section or right click on the desired model type.

## Example Applications / Downstream Tasks

- Named Entity Recognition (NER)


## Setup

It is recommended to run all commands within a virtual environment.

- Basic Setup


    pip install -r requirements.txt  # install required packages
    bash setup_apex.sh               # installs apex from https://github.com/NVIDIA/apex
    
        
- Download the models to directory `./pretrained_models`:


    bash setup_pretrained_models.sh --all          # use --base or --large if you only want to download one model
    
    
- Optional: Download the NER dataset to directory `./datasets`:


    bash setup_datasets.sh --all                   # use --ner if you only want to download one dataset


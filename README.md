# nlp-bert

BERT and other transformer models are available in different
languages from the transformers package. 

Arbetsförmedlingen (Swedish Public Employment Service) has developed 
Swedish BERT models which were trained on Swedish Wikipedia with approximatelly 2 million articles and 300 million words.

This repository provides 

- `Models`: Arbetsförmedlingens Swedish BERT models
- `Downstream Task Support`: Tools for downstream task training on publicly available Swedish datasets

  
  
## Models

### Usage

The Swedish BERT models are provided in the format used by the transformers package. 

This means that both tokenizer and model can be instantiated using the `from_pretrained()` method 
of the BERT-related transformers classes like so:

    tokenizer = BertTokenizer.from_pretrained(<model type>)
    
    model = BertModel.from_pretrained(<model type>)
    model = BertForTokenClassification.from_pretrained(<model type>) 
    ..

### Available Model Types  
  
- `swe-bert-base`

    12-layer, 768-hidden, 12-heads, 110M parameters --- size: ~1.5 GB zipped, ~1.7 GB unzipped
- `swe-bert-large`

    24-layer, 1024-hidden, 16-heads, 340M parameters --- size: ~4.6 GB zipped, ~5.0 GB unzipped
    
For download see the "Setup" section or right click on the desired model type.

  
  
## Downstream Task Support

- Named Entity Recognition (NER)

  
    
## Getting Started

It is recommended to run all commands and notebooks within a virtual environment.

### Setup

- Install packages:


    pip install -r requirements.txt
    bash setup_apex.sh                            # installs apex from https://github.com/NVIDIA/apex
    
        
- Download models to directory `./pretrained_models`:


    bash setup_pretrained_models.sh (base/large)  # use base or large if you only want to download one model
    
    
- Optional: Download datasets to directory `./datasets`:


    bash setup_datasets.sh (ner)                  # use ner if you only want to download one dataset


### Notebooks

- `bert.ipynb` shows how to get started with the Swedish BERT models
- `bert_ner.ipynb` shows how to train a Swedish BERT model on the 
NER downstream task using the publicly available Swedish NER corpus dataset.  

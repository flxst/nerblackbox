# Datasets and Models

**nerblackbox** and its [Command Line Interface](../cli) & [Python API](../python_api) 
work out of the box (see [Getting Started](../getting_started)) 
for **built-in** datasets and models.
**Custom** datasets and models can easily be included.

-----------
## Built-in Datasets 

| Name               | Language | Open Source | Sample Type  | #Samples (Train, Val, Test) | Directory Name     | Required Files | Source |               
|---                 |---       |---          |---           |---                          |---                 |---             |---     |
| CoNLL 2003         | English  | Yes         | Sentence     | (14040, 3249, 3452)         | conll2003          | ---            | [Description](https://www.clips.uantwerpen.be/conll2003/ner/); [Data](https://github.com/patverga/torch-ner-nlp-from-scratch/tree/master/data/conll2003)   |
| Swedish NER Corpus | Swedish  | Yes         | Sentence     | (4819, 2065, 2452)          | swedish_ner_corpus | ---            | [Description+Data](https://github.com/klintan/swedish-ner-corpus)   |
| SIC                | Swedish  | Yes         | Sentence     | (437, 187, 268)             | sic                | ---            | [Description+Data](https://www.ling.su.se/english/nlp/corpora-and-resources/sic)   |
| SUC 3.0            | Swedish  | No          | Sentence     | (71046, 1546, 1568)         | suc                | `suc-*.conll`  | [Description](https://www.ling.su.se/english/nlp/corpora-and-resources/suc)   |

- Datasets are pre-processed and made available by:

    !!! note "Open Source"
        === "CLI"
            ``` bash
            nerbb download    
            ```
        === "Python"
            ``` python
            nerbb.download()  
            ```
    !!! warning "Not Open Source"
        - create folder: `mkdir ./data/datasets/<Directory Name>`
        - move `Required Files` manually to `./data/datasets/<Directory Name>`
        - set up dataset:
              
        === "CLI"
            ``` bash
            nerbb set_up_dataset <Directory Name>
            ```
        === "Python"
            ``` python
            nerbb.set_up_dataset(<Directory Name>)  
            ```
  
- Additional dataset details (tags, tag distribution, ..) can be found in `./data/datasets/<Directory Name>/analyze_data`

-----------
## Built-in Models

- All built-in or community-uploaded BERT models of the [transformers library](https://huggingface.co/transformers/)

-----------
## Custom Datasets

To include your own custom dataset, do the following:

- Create a folder ``./data/datasets/<new_dataset>`` with the following files:

    - ``train.csv``
    - ``val.csv``
    - ``test.csv``
- Each row of the respective ``*.csv`` files has to contain one training sample in the format
  ``<labels> <tab> <text>``,
  e.g. ``0 0 0 0 0 0 PER <tab> this is a sample with a person``

- Use ``dataset_name = <new_dataset>`` in your experiment configuration file.

<!---
TODO
Own custom datasets can also be created programmatically (like the :ref:`Built-in datasets <builtindatasets>`):
- (todo: revise the following)
- Create a new module ``./data/datasets/formatter/<new_dataset>_formatter.py``
- Derive the class ``<NewDataset>Formatter`` from ``BaseFormatter`` and implement the abstract base methods
- (todo: additional instructions needed here)
--->

-----------
## Custom Models

To include your own custom model, do the following:

- Create a new folder ``./data/pretrained_models/<new_model>`` with the following files:

    - ``config.json``
    - ``pytorch_model.bin``
    - ``vocab.txt``

- ``<new_model>`` must include the architecture type, e.g. ``bert``

- Use ``pretrained_model_name = <new_model>`` in your experiment configuration file.

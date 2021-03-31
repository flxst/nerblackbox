# Datasets and Models

## Built-in Datasets & Models

**nerblackbox** and its [Command Line Interface](../cli) & [Python API](../python_api) work out of the box (see [Getting Started](../getting_started)) for the built-in datasets and models.


- Built-in datasets:

    | Name               | Language | Open Source | Directory Name     | Required Files |                
    |---                 |---       |---          |---                 |---             |
    | CoNLL 2003         | English  | Yes         | conll2003          | ---            |
    | Swedish NER Corpus | Swedish  | Yes         | swedish_ner_corpus | ---            |
    | SIC                | Swedish  | Yes         | sic                | ---            |
    | SUC 3.0            | Swedish  | No          | suc                | `suc-train.conll`, `suc-dev.conll`, `suc-test.conll`                |

    Datasets are pre-processed and made available by:
  
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
  
- Built-in models:

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

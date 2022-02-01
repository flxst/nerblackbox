# Datasets and Models

**nerblackbox** and its [Python API](../../python_api/overview) & [CLI](../../cli/cli)
work out of the box (see [Getting Started](../getting_started)) 
for **built-in** datasets and models.
**Custom** datasets and models can easily be included.

-----------
## Built-in Datasets 

- All [community-uploaded datasets](https://huggingface.co/datasets) of the [datasets library](https://huggingface.co/docs/datasets/)
- The following built-in datasets:

    | Name               | Language | Open Source | Annotation Scheme | Sample Type  | #Samples (Train, Val, Test) | Directory Name     | Required Files | Source |               
    |---                 |---       |---          |---                |---           |---                          |---                 |---             |---     |
    | CoNLL 2003         | English  | Yes         | BIO               | Sentence     | (14041, 3250, 3453)         | conll2003          | ---            | [Description](https://www.clips.uantwerpen.be/conll2003/ner/); [Data](https://github.com/patverga/torch-ner-nlp-from-scratch/tree/master/data/conll2003)   |
    | Swedish NER Corpus | Swedish  | Yes         |  IO               | Sentence     | (4819, 2066, 2453)          | swedish_ner_corpus | ---            | [Description+Data](https://github.com/klintan/swedish-ner-corpus)   |
    | SIC                | Swedish  | Yes         | BIO               | Sentence     | (436, 188, 268)             | sic                | ---            | [Description+Data](https://www.ling.su.se/english/nlp/corpora-and-resources/sic)   |
    | SUC 3.0            | Swedish  | No          | BIO               | Sentence     | (71046, 1546, 1568)         | suc                | `suc-*.conll`  | [Description](https://www.ling.su.se/english/nlp/corpora-and-resources/suc)   |
    | Swe-NERC           | Swedish  | Yes         | BIO               | Sentence     | (6841, 878, 891)            | swe_nerc           | ---            | [Description](https://gubox.app.box.com/v/SLTC-2020-paper-17); [Data](https://spraakbanken.gu.se/lb/resurser/swe-nerc/)   |

    - These datasets are pre-processed and made available by:

        ??? note "Open Source"
            === "CLI"
                ``` bash
                nerbb download    
                ```
            === "Python"
                ``` python
                nerbb.download()  
                ```
        ??? warning "Not Open Source"
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

- All [built-in or community-uploaded models](https://huggingface.co/models) of the [transformers library](https://huggingface.co/transformers/)
  that use the [WordPiece Tokenizer](https://huggingface.co/transformers/tokenizer_summary.html#wordpiece), e.g.
    - BERT
    - DistilBERT
    - Electra

-----------
## Custom Datasets

To include your own custom dataset, do the following:

- Create a folder ``./data/datasets/<custom_dataset>``
    and add three files ``train.*``, ``val.*``, ``test.*`` to it. 
    The filename extension is either ``* = jsonl`` or ``* = csv``, depending on the [data format](../../features/support_pretokenized) (standard or pretokenized).

- If your data consists of standard annotations, it must adhere to the following ``.jsonl`` format:
      ```
      {"text": "President Barack Obama went to Harvard", "tags": [{"token": "President Barack Obama", "tag": "PER", "char_start": 0, "char_end": 22}, {"token": "Harvard", "tag": "ORG", "char_start": 31, "char_end": 38}}
      ```
      Each row has to contain a single training sample in the format
      ``{"text": str, "tags": List[Dict]}``, where ``Dict = {"token": str, "tag": str, "char_start": int, "char_end": int}``
      This format is commonly used by annotation tools.
- If your data consists of pretokenized annotations, it must adhere to the following ``.csv`` format:
      ```
      PER PER PER O O ORG <tab> President Barack Obama went to Harvard
      ```
      Each row has to contain a single training sample in the format
      ``<tags> <tab> <text>``, where in ``<tags>`` and ``<text>`` the tags and tokens are separated by whitespace.
      This format is suitable for many public datasets. 

- Use ``dataset_name = <custom_dataset>`` as [`parameter`](../parameters/#1-dataset) 
    when [`fine-tuning a model`](../getting_started/#3-fine-tune-a-model).

<!---
TODO
Own custom datasets can also be created programmatically (like the :ref:`Built-in datasets <builtindatasets>`):
- (todo: revise the following)
- Create a new module ``./data/datasets/formatter/<custom_dataset>_formatter.py``
- Derive the class ``<NewDataset>Formatter`` from ``BaseFormatter`` and implement the abstract base methods
- (todo: additional instructions needed here)
--->

-----------
## Custom Models

To include your own custom model, do the following:

- Create a new folder ``./data/pretrained_models/<custom_model>`` with the following files:

    - ``config.json``
    - ``pytorch_model.bin``
    - ``vocab.txt``

- ``<custom_model>`` must include the architecture type, e.g. ``bert``

- Use ``pretrained_model_name = <custom_model>`` as [parameter](../parameters/#2-model)
    when [`fine-tuning a model`](../getting_started/#3-fine-tune-a-model).

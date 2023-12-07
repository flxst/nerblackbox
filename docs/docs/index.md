# Start

**nerblackbox** - A High-level Library for Named Entity Recognition in Python


Latest version: 1.0.0

-----------
## Resources

* Source Code: [https://github.com/flxst/nerblackbox](https://github.com/flxst/nerblackbox)
* Documentation: [https://flxst.github.io/nerblackbox](https://flxst.github.io/nerblackbox)
* Paper: [https://aclanthology.org/2023.nlposs-1.20](https://aclanthology.org/2023.nlposs-1.20)
* PyPI: [https://pypi.org/project/nerblackbox](https://pypi.org/project/nerblackbox)

-----------
## Installation

``` bash
pip install nerblackbox
```

-----------
## About

![nerblackboxmain](images/nerblackbox_sources.png#main)

Take a dataset from one of many available sources.
Then train, evaluate and apply a language model 
in a few simple steps.

=== "Data"

    - Choose a dataset from **HuggingFace (HF)**, the **Local Filesystem (LF)**, or a **Built-in (BI)** dataset
    ``` python
    dataset = Dataset("conll2003",  source="HF")  # HuggingFace
    dataset = Dataset("my_dataset", source="LF")  # Local Filesystem
    dataset = Dataset("swe_nerc",   source="BI")  # Built-in

    ```

    - Set up the dataset
    ``` python
    dataset.set_up()
    ```
    &nbsp;

    Datasets from an **Annotation Tool (AT)** server can also be used. See [Data](data.md) for more details.

=== "Training"
    
    - Define the training by choosing a pretrained model and a dataset
    ``` python
    training = Training("my_training", model="bert-base-cased", dataset="conll2003")
    ```

    - Run the training and get the performance of the fine-tuned model
    ``` python
    training.run()
    training.get_result(metric="f1", level="entity", phase="test")
    # 0.9045
    ```
    &nbsp;

    See [Training](training.md) for more details.
=== "Evaluation"

    - Load the model
    ```python
    model = Model.from_training("my_training")
    ```

    - Evaluate the model
    ```python
    results = model.evaluate_on_dataset("conll2003", phase="test")
    results["micro"]["entity"]["f1"]
    # 0.9045
    ```
    &nbsp;

    See [Evaluation](evaluation.md) for more details.
=== "Inference"

    - Load the model
    ``` python
    model = Model.from_training("my_training")
    ```

    - Let the model predict
    ``` python
    model.predict("The United Nations has never recognised Jakarta's move.")  
    # [[
    #  {'char_start': '4', 'char_end': '18', 'token': 'United Nations', 'tag': 'ORG'},
    #  {'char_start': '40', 'char_end': '47', 'token': 'Jakarta', 'tag': 'LOC'}
    # ]]
    ```

    See [Inference](inference.md) for more details.


-----------
## Get Started

In order to get familiar with **nerblackbox**, it is recommended to 

1. read the doc sections 
[Preparation](preparation.md), 
[Data](data.md),
[Training](training.md),
[Evaluation](evaluation.md) and
[Inference](inference.md)

2. go through one of the example [notebooks](https://github.com/flxst/nerblackbox/tree/master/notebooks) 

3. check out the [Python API documentation](python_api/overview.md)

-----------
## Features

*Data*

* Integration of Datasets from Multiple Sources (HuggingFace, Annotation Tools, ..)
* Support for Multiple Dataset Types (Standard, Pretokenized)
* Support for Multiple Annotation Schemes (IO, BIO, BILOU)
* Text Encoding

*Training*

* Adaptive Fine-tuning
* Hyperparameter Search
* Multiple Runs with Different Random Seeds
* Detailed Analysis of Training Results

*Evaluation*

* Evaluation of Any Model on Any Dataset

*Inference*

* Versatile Model Inference (Entity/Word Level, Probabilities, ..)

*Other*

* Full Compatibility with HuggingFace
* GPU Support
* Language Agnosticism


-----------
## Citation

``` tex
@inproceedings{stollenwerk-2023-nerblackbox,
    title = "nerblackbox: A High-level Library for Named Entity Recognition in Python",
    author = "Stollenwerk, Felix",
    editor = "Tan, Liling  and
      Milajevs, Dmitrijs  and
      Chauhan, Geeticka  and
      Gwinnup, Jeremy  and
      Rippeth, Elijah",
    booktitle = "Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023)",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Empirical Methods in Natural Language Processing",
    url = "https://aclanthology.org/2023.nlposs-1.20",
    pages = "174--178",
    abstract = "We present **nerblackbox**, a python library to facilitate the use of state-of-the-art transformer-based models for named entity recognition. It provides simple-to-use yet powerful methods to access data and models from a wide range of sources, for fully automated model training and evaluation as well as versatile model inference. While many technical challenges are solved and hidden from the user by default, **nerblackbox** also offers fine-grained control and a rich set of customizable features. It is thus targeted both at application-oriented developers as well as machine learning experts and researchers.",
}
```

-----------
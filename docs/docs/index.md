# Start

**nerblackbox** - a high-level library for named entity recognition in python

latest version: 0.0.15

-----------
## Resources

* source code: [https://github.com/flxst/nerblackbox](https://github.com/flxst/nerblackbox)
* documentation: [https://flxst.github.io/nerblackbox](https://flxst.github.io/nerblackbox)
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
    
    - Define a fine-tuning experiment by choosing a pretrained model and a dataset
    ``` python
    experiment = Experiment("my_experiment", model="bert-base-cased", dataset="conll2003")
    ```

    - Run the experiment and get the performance of the fine-tuned model
    ``` python
    experiment.run()
    experiment.get_result(metric="f1", level="entity", phase="test")
    # 0.9045
    ```
    &nbsp;

    See [Training](training.md) for more details.
=== "Evaluation"

    - Load the model
    ```python
    model = Model.from_experiment("my_experiment")
    ```

    - Evaluate the model
    ```python
    evaluation_dict = model.evaluate_on_dataset("ehealth_kd", phase="test")
    evaluation_dict["micro"]["entity"]["f1"]
    # 0.9045
    ```
    &nbsp;

    See [Evaluation](evaluation.md) for more details.
=== "Inference"

    - Load the model
    ``` python
    model = Model.from_experiment("my_experiment")
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
@misc{nerblackbox,
  author = {Stollenwerk, Felix},
  title  = {nerblackbox: a high-level library for named entity recognition in python},
  year   = {2021},
  url    = {https://github.com/flxst/nerblackbox},
}
```

-----------
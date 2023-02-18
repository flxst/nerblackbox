# Overview

**nerblackbox** - a python package to fine-tune transformer-based language models for named entity recognition (NER).

latest version: 0.0.14

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

![nerblackbox overview diagram](images/nerblackbox.png)

Fine-tune a [language model](https://huggingface.co/transformers/pretrained_models.html) for
[named entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) in a few simple steps:

   1. Define a fine-tuning experiment by choosing a pretrained model and a dataset
       ``` python
       experiment = Experiment("my_experiment", model="bert-base-cased", dataset="conll2003")
       ```
   

   2. Run the experiment and get the performance of the fine-tuned model
       ``` python
       experiment.run()
       experiment.get_result(metric="f1", level="entity", phase="test")
       # 0.9045
       ```

   3. Use the fine-tuned model for inference
       ``` python
       model = Model.from_experiment("my_experiment")
       model.predict("The United Nations has never recognised Jakarta's move.")  
       # [[
       #  {'char_start': '4', 'char_end': '18', 'token': 'United Nations', 'tag': 'ORG'},
       #  {'char_start': '40', 'char_end': '47', 'token': 'Jakarta', 'tag': 'LOC'}
       # ]]
       ```

There is much more to it than that! See [Usage](usage/getting_started) to get started.

-----------
## Features

*Data*

* Support for Different Data Formats
* Support for Different Annotation Schemes
* Integration of HuggingFace Datasets
* Text Encoding

*Training*

* Adaptive Fine-tuning
* Hyperparameter Search
* Multiple Runs with Different Random Seeds
* Detailed Analysis of Training Results

*Evaluation*

* Evaluation of a Model on a Dataset

*Inference*

* Versatile Model Inference

*Other*

* Compatibility with HuggingFace
* GPU Support
* Language Agnosticism

See [Features](features/overview) for more details.


-----------
## Citation

``` tex
@misc{nerblackbox,
  author = {Stollenwerk, Felix},
  title  = {nerblackbox: a python package to fine-tune transformer-based language models for named entity recognition},
  year   = {2021},
  url    = {https://github.com/flxst/nerblackbox},
}
```

-----------
===========
nerblackbox
===========

A python package to fine-tune transformer-based language models for named entity recognition (NER).

.. image:: https://img.shields.io/pypi/v/nerblackbox
    :target: https://pypi.org/project/nerblackbox
    :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/nerblackbox
    :target: https://www.python.org/doc/versions/
    :alt: PyPI - Python Version

.. image:: https://github.com/flxst/nerblackbox/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/flxst/nerblackbox/actions/workflows/python-package.yml
    :alt: CI

.. image:: https://coveralls.io/repos/github/flxst/nerblackbox/badge.svg?branch=master
    :target: https://coveralls.io/github/flxst/nerblackbox?branch=master

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/pypi/l/nerblackbox
    :target: https://github.com/flxst/nerblackbox/blob/latest/LICENSE.txt
    :alt: PyPI - License

Resources
=========

- Source Code: https://github.com/flxst/nerblackbox
- Documentation: https://flxst.github.io/nerblackbox
- PyPI: https://pypi.org/project/nerblackbox

Installation
============

::

    pip install nerblackbox

About
=====

.. image:: https://raw.githubusercontent.com/flxst/nerblackbox/master/docs/docs/images/nerblackbox_sources.png

Take a dataset from one of many available sources.
Then train, evaluate and apply a language model
in a few simple steps.

1. Data
"""""""

- Choose a dataset from **HuggingFace (HF)**, the **Local Filesystem (LF)**, an **Annotation Tool (AT)** server, or a **Built-in (BI)** dataset

::

    dataset = Dataset("conll2003",  source="HF")  # HuggingFace
    dataset = Dataset("my_dataset", source="LF")  # Local Filesystem
    dataset = Dataset("swe_nerc",   source="BI")  # Built-in

- Set up the dataset

::

    dataset.set_up()


2. Training
"""""""""""

- Define a fine-tuning experiment by choosing a pretrained model and a dataset

::

    experiment = Experiment("my_experiment", model="bert-base-cased", dataset="conll2003")

- Run the experiment and get the performance of the fine-tuned model

::

    experiment.run()
    experiment.get_result(metric="f1", level="entity", phase="test")
    # 0.9045

3. Evaluation
^^^^^^^^^^^^^

- Load the model

::

    model = Model.from_experiment("my_experiment")

- Evaluate the model

::

    evaluation_dict = model.evaluate_on_dataset("ehealth_kd", "jsonl", phase="test")
    evaluation_dict["micro"]["entity"]["f1"]
    # 0.9045


4. Inference
^^^^^^^^^^^^

- Load the model

::

    model = Model.from_experiment("my_experiment")

- Let the model predict

::

    model.predict("The United Nations has never recognised Jakarta's move.")
    # [[
    #  {'char_start': '4', 'char_end': '18', 'token': 'United Nations', 'tag': 'ORG'},
    #  {'char_start': '40', 'char_end': '47', 'token': 'Jakarta', 'tag': 'LOC'}
    # ]]

There is much more to it than that! See the `documentation <https://flxst.github.io/nerblackbox/usage/getting_started/>`__ to get started.

Features
========

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

See the `documentation <https://flxst.github.io/nerblackbox/features/overview>`__ for more details.

Citation
========

::

    @misc{nerblackbox,
      author = {Stollenwerk, Felix},
      title  = {nerblackbox: a python package to fine-tune transformer-based language models for named entity recognition},
      year   = {2021},
      url    = {https://github.com/flxst/nerblackbox},
    }

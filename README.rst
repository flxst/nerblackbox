===========
nerblackbox
===========

A High-level Library for Named Entity Recognition in Python

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
- Paper: https://aclanthology.org/2023.nlposs-1.20
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

- Define the training by choosing a pretrained model and a dataset

::

    training = Training("my_training", model="bert-base-cased", dataset="conll2003")

- Run the training and get the performance of the fine-tuned model

::

    training.run()
    training.get_result(metric="f1", level="entity", phase="test")
    # 0.9045


3. Evaluation
"""""""""""""

- Load the model

::

    model = Model.from_training("my_training")

- Evaluate the model

::

    results = model.evaluate_on_dataset("ehealth_kd", phase="test")
    results["micro"]["entity"]["f1"]
    # 0.9045


4. Inference
""""""""""""

- Load the model

::

    model = Model.from_training("my_training")

- Let the model predict

::

    model.predict("The United Nations has never recognised Jakarta's move.")
    # [[
    #  {'char_start': '4', 'char_end': '18', 'token': 'United Nations', 'tag': 'ORG'},
    #  {'char_start': '40', 'char_end': '47', 'token': 'Jakarta', 'tag': 'LOC'}
    # ]]

There is much more to it than that! See the `documentation <https://flxst.github.io/nerblackbox>`__ to get started.

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

See the `documentation <https://flxst.github.io/nerblackbox>`__ for details.

Citation
========

::

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

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

.. image:: https://raw.githubusercontent.com/flxst/nerblackbox/master/docs/docs/images/nerblackbox.png

Fine-tune a `language model <https://huggingface.co/transformers/pretrained_models.html>`_ for
`named entity recognition <https://en.wikipedia.org/wiki/Named-entity_recognition>`_ in a few simple steps:

1. Define a fine-tuning experiment by choosing a pretrained model and a dataset

::

   experiment = Experiment("my_experiment", model="bert-base-cased", dataset="conll2003")


2. Run the experiment and get the performance of the fine-tuned model

::

   experiment.run()
   experiment.get_result(metric="f1", level="entity", phase="test")
   # 0.9045

3. Use the fine-tuned model for inference

::

    model = Model.from_experiment("my_experiment")
    model.predict("The United Nations has never recognised Jakarta's move.")
    # [[
    #  {'char_start': '4', 'char_end': '18', 'token': 'United Nations', 'tag': 'ORG'},
    #  {'char_start': '40', 'char_end': '47', 'token': 'Jakarta', 'tag': 'LOC'}
    # ]]

There is much more to it than that! See the `documentation <https://flxst.github.io/nerblackbox/usage/getting_started/>`_ to get started.

Features
========

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

See the `documentation <https://flxst.github.io/nerblackbox/features/overview>`_ for more details.

Citation
========

::

    @misc{nerblackbox,
      author = {Stollenwerk, Felix},
      title  = {nerblackbox: a python package to fine-tune transformer-based language models for named entity recognition},
      year   = {2021},
      url    = {https://github.com/flxst/nerblackbox},
    }

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

.. image:: https://travis-ci.org/af-ai-center/nerblackbox.svg?branch=master
    :target: https://travis-ci.com/github/af-ai-center/nerblackbox
    :alt: Travis CI

.. image:: https://coveralls.io/repos/github/af-ai-center/nerblackbox/badge.svg?branch=master
    :target: https://coveralls.io/github/af-ai-center/nerblackbox?branch=master

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/pypi/l/nerblackbox
    :target: https://github.com/af-ai-center/nerblackbox/blob/latest/LICENSE.txt
    :alt: PyPI - License

Resources
=========

- Source Code: https://github.com/af-ai-center/nerblackbox
- Documentation: https://af-ai-center.github.io/nerblackbox
- PyPI: https://pypi.org/project/nerblackbox

About
=====

`Transformer-based language models <https://arxiv.org/abs/1706.03762>`_ like `BERT <https://arxiv.org/abs/1810.04805>`_ have had a `game-changing impact <https://paperswithcode.com/task/language-modelling>`_ on Natural Language Processing.

In order to utilize `Hugging Face's publicly accessible pretrained models <https://huggingface.co/transformers/pretrained_models.html>`_ for
`Named Entity Recognition <https://en.wikipedia.org/wiki/Named-entity_recognition>`_,
one needs to retrain (or "fine-tune") them using labeled text.

**nerblackbox makes this easy.**

.. image:: https://raw.githubusercontent.com/af-ai-center/nerblackbox/master/docs/docs/images/nerblackbox.png

You give it

- a **Dataset** (labeled text)
- a **Pretrained Model** (transformers)

and you get

- the best **Fine-tuned Model**
- its **Performance** on the dataset

Installation
============

::

    pip install nerblackbox

Usage
=====

see documentation: https://af-ai-center.github.io/nerblackbox

Citation
========

::

    @misc{nerblackbox,
      author = {Stollenwerk, Felix},
      title  = {nerblackbox: a python package to fine-tune transformer-based language models for named entity recognition},
      year   = {2021},
      url    = {https://github.com/af-ai-center/nerblackbox},
    }

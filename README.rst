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

About
=====

`Transformer-based language models <https://arxiv.org/abs/1706.03762>`_ like `BERT <https://arxiv.org/abs/1810.04805>`_ have had a `game-changing impact <https://paperswithcode.com/task/language-modelling>`_ on Natural Language Processing.

In order to utilize `Hugging Face's publicly accessible pretrained models <https://huggingface.co/transformers/pretrained_models.html>`_ for
`Named Entity Recognition <https://en.wikipedia.org/wiki/Named-entity_recognition>`_,
one needs to retrain (or "fine-tune") them using labeled text.

**nerblackbox makes this easy.**

.. image:: https://raw.githubusercontent.com/flxst/nerblackbox/master/docs/docs/images/nerblackbox.png

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

see documentation: https://flxst.github.io/nerblackbox

Citation
========

::

    @misc{nerblackbox,
      author = {Stollenwerk, Felix},
      title  = {nerblackbox: a python package to fine-tune transformer-based language models for named entity recognition},
      year   = {2021},
      url    = {https://github.com/flxst/nerblackbox},
    }

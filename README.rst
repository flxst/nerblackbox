===========
nerblackbox
===========

A python package to smoothly fine-tune transformer-based models for Named Entity Recognition.

About
=====

`Transformer-based models <https://arxiv.org/abs/1706.03762>`_ like `BERT <https://arxiv.org/abs/1810.04805>`_ have had a `game-changing impact <https://paperswithcode.com/task/language-modelling>`_ on Natural Language Processing.

In order to utilize the `publicly accessible pretrained models <https://huggingface.co/transformers/pretrained_models.html>`_ for
`Named Entity Recognition <https://en.wikipedia.org/wiki/Named-entity_recognition>`_,
one needs to retrain (or "fine-tune") them using labeled text.

**nerblackbox makes this easy.**

.. image:: ./docs/_static/nerblackbox.png

You give it

- a **Dataset** (labeled text)
- a **Pretrained Model** (transformers)

and you get

- the best **Fine-tuned Model**
- its **Performance** on the dataset


Installation
============

    ``pip install nerblackbox``

Usage
=====

Fine-tuning can be done in a few simple steps using an "experiment configuration file"

.. code-block:: python

   # cat <experiment_name>.ini
   dataset_name = swedish_ner_corpus
   pretrained_model_name = af-ai-center/bert-base-swedish-uncased

and either the Command Line Interface (CLI) or the Python API:

.. code-block:: python

   # CLI
   nerbb run_experiment <experiment_name>
   nerbb predict <experiment_name> <text_input>

   # Python API
   nerbb = NerBlackBox()
   nerbb.run_experiment(<experiment_name>)
   nerbb.predict(<experiment_name>, <text_input>)

Documentation
=============

see https://nerblackbox.readthedocs.io
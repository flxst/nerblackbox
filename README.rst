
.. .. include:: ./docs_source/source/shared/main1.rst

===========
nerblackbox
===========

A python package to smoothly fine-tune transformer-based models for Named Entity Recognition (NER).

Resources
=========

- Source Code: https://github.com/af-ai-center/nerblackbox
- Documentation: https://af-ai-center.github.io/nerblackbox
- PyPI: https://pypi.org/project/nerblackbox

About
=====

`Transformer-based models <https://arxiv.org/abs/1706.03762>`_ like `BERT <https://arxiv.org/abs/1810.04805>`_ have had a `game-changing impact <https://paperswithcode.com/task/language-modelling>`_ on Natural Language Processing.

In order to utilize the `publicly accessible pretrained models <https://huggingface.co/transformers/pretrained_models.html>`_ for
`Named Entity Recognition <https://en.wikipedia.org/wiki/Named-entity_recognition>`_,
one needs to retrain (or "fine-tune") them using labeled text.

**nerblackbox makes this easy.**

.. image:: https://raw.githubusercontent.com/af-ai-center/nerblackbox/master/docs/_static/nerblackbox.png

.. .. include:: ./docs_source/source/shared/main2.rst

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

.. .. include:: ./docs_source/source/shared/usage.rst

Fine-tuning can be done in a few simple steps using an "experiment configuration file"

.. code-block:: python

   # cat <experiment_name>.ini
   dataset_name = swedish_ner_corpus
   pretrained_model_name = af-ai-center/bert-base-swedish-uncased

and either the Command Line Interface (CLI) or the Python API:

.. code-block:: python

   # CLI
   nerbb run_experiment <experiment_name>          # fine-tune
   nerbb get_experiment_results <experiment_name>  # get results/performance
   nerbb predict <experiment_name> <text_input>    # apply best model

   # Python API
   nerbb = NerBlackBox()
   nerbb.run_experiment(<experiment_name>)         # fine-tune
   nerbb.get_experiment_results(<experiment_name>) # get results/performance
   nerbb.predict(<experiment_name>, <text_input>)  # apply best model

Datasets and Models
===================

Content:

- :ref:`builtindatasetsandmodels`
- :ref:`customdatasets`
- :ref:`custommodels`


.. _builtindatasetsandmodels:

Built-in Datasets & Models
--------------------------
**nerblackbox** and its CLI/API (see :ref:`quickstart` and :ref:`apidocumentation`) work out of the box for the built-in datasets and models.

.. _builtindatasets:

- Built-in datasets:

    - CoNLL2003 (english)
    - Swedish NER Corpus (swedish)

- Built-in models:

    - All built-in or community-uploaded BERT models of the `transformers library <https://huggingface.co/transformers/>`_


.. _customdatasets:

Custom Datasets
---------------

To include your own custom dataset, do the following:

- Create a folder ``./data/datasets/<new_dataset>`` with the following files:

    - ``train.csv``
    - ``val.csv``
    - ``test.csv``
- Each row of the respective ``*.csv`` files has to contain one training sample in the format
    ``<labels> <tab> <text>``,
    e.g. ``0 0 0 0 0 0 PER <tab> this is a sample with a person``

.. TODO
 Own custom datasets can also be created programmatically (like the :ref:`Built-in datasets <builtindatasets>`):
 - (todo: revise the following)
 - Create a new module ``./data/datasets/formatter/<new_dataset>_formatter.py``
 - Derive the class ``<NewDataset>Formatter`` from ``BaseFormatter`` and implement the abstract base methods
 - (todo: additional instructions needed here)


.. _custommodels:

Custom Models
-------------

To include your own custom model, do the following:

 - Create a new folder ``./data/pretrained_model/<new_model>`` with the following files:

    - ``config.json``
    - ``pytorch_model.bin``
    - ``vocab.txt``

  The folder name must include the architecture type, e.g. ``bert``

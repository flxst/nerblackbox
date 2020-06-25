.. nerblackbox documentation master file, created by
   sphinx-quickstart on Thu May 28 10:07:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========
nerblackbox
===========

A python package to seamlessly fine-tune transformer models for Named Entity Recognition

.. image:: _static/nerblackbox.png
  :width: 600
  :alt: NER Black Box Overview Diagram

You give it

- a **Dataset** (labeled text)
- a **Pretrained Model** (transformers)

and you get

- the best **Fine-tuned Model**
- its **Performance** on the dataset

Usage
----------------------------

Fine-tuning can be done in a few simple steps using either the Command Line Interface (CLI) or the Python API:

.. code-block:: python

   # CLI
   nerbb --run_experiment <experiment_name>
   nerbb --predict <experiment_name> <text_input>

   # Python API
   nerbb = NerBlackBox()
   nerbb.run_experiment(<experiment_name>)
   nerbb.predict(<experiment_name>, <text_input>)

See :ref:`quickstart` for more details.

Some features of nerblackbox
----------------------------

* GPU support
* Hyperparameter Search
* Early Stopping

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   datasetsandmodels
   nerblackbox

.. Indices and tables
   ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

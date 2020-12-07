
.. _quickstart:

Quickstart
==========

**nerblackbox** allows to fine-tune a transformer model in a few simple steps,
using either the

* the :ref:`cli` (command line interface)

    .. code-block:: console

        nerbb --help

* or the :ref:`api`.

    .. code-block:: python

        from nerblackbox import NerBlackBox
        nerbb = NerBlackBox()


1. Initialization
-----------------

The following command needs to be executed once:

.. code-block:: python

    # CLI
    nerbb init

    # Python API
    nerbb.init()


It creates a ``./data`` directory with the following structure:

.. code-block:: xml

    data/
    └── datasets
        └── conll2003           # built-in dataset (english)
            └── train.csv
            └── val.csv
            └── test.csv
        └── swedish_ner_corpus  # built-in dataset (swedish)
            └── train.csv
            └── val.csv
            └── test.csv
    └── experiment_configs
        └── default.ini
        └── exp_test.ini
    └── pretrained_models       # custom model checkpoints
    └── results



2. Single Experiment
--------------------

2a. Define an experiment
^^^^^^^^^^^^^^^^^^^^^^^^

A single experiment is defined through a configuration file in ``./data/experiment_configs/<experiment_name>.ini``

    * See ``<experiment_name> = exp_test`` for an example

    * The following parameters need to be specified in the configuration file:
        * dataset
        * pretrained model
        * hyperparameters

    * An experiment can entail multiple training runs with different hyperparameter combinations (manual search).

    * One can view an experiment configuration as follows:

        .. code-block:: python

            # CLI
            nerbb show_experiment_config <experiment_name>

            # Python API
            nerbb.show_experiment_config(<experiment_name>)

2b. Run an experiment
^^^^^^^^^^^^^^^^^^^^^

Once a single experiment is defined, the following command can be used to run it.

    .. code-block:: python

        # CLI
        nerbb run_experiment <experiment_name>

        # Python API
        nerbb.run_experiment(<experiment_name>)

2c. Get experiment results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once an experiment is finished, one can inspect the main results or detailed results:

    * Get main results:

        .. code-block:: python

            # CLI
            nerbb get_experiment_results <experiment_name>  # prints overview on runs

            # Python API
            experiment_results = nerbb.get_experiment_results(<experiment_name>)

        See :ref:`experimentresults`
        for details on how to use ``experiment_results``

    * Get detailed results & run histories using either mlflow or tensorboard:

        .. code-block:: python

            # CLI
            nerbb mlflow       # + enter http://localhost:5000 in your browser
            nerbb tensorboard  # + enter http://localhost:6006 in your browser


2d. Predict tags using the best model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code-block:: python

        # e.g. <text_input> = 'some text that needs to be tagged'

        # CLI
        nerbb predict <experiment_name> <text_input>

        # Python API
        nerbb.predict(<experiment_name>, <text_input>)

        # Python API (w/o having to reload the best model for multiple predictions)
        experiment_results = nerbb.get_experiment_results(<experiment_name>)
        experiment_results.best_model.predict(<text_input>)

    See :ref:`nermodelpredict`
    for details on how to use ``experiments_results.best_model``

3. Multiple Experiments
-----------------------

Once one or more experiments have been run, the following commands can be used to access their results:

    a. Get Experiments Overview

        .. code-block:: python

            # CLI
            nerbb get_experiments

            # Python API
            nerbb.get_experiments()

    b. Get Best Runs Overview:

        .. code-block:: python

            # CLI
            nerbb get_experiments_results

            # Python API
            experiments_results = nerbb.get_experiments_results()

        See :ref:`experimentsresults`
        for details on how to use ``experiments_results``

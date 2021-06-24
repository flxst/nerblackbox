# Getting Started

Use either the [`CLI (Command Line Interface)`](../../cli/cli) or the [`Python API`](../../python_api/overview).

??? note "basic usage"
    === "CLI"

        ``` bash
        nerbb --help
        ```

    === "Python"

        ``` python
        from nerblackbox import NerBlackBox
        nerbb = NerBlackBox()
        ```

-----------
## 1. Initialization

The following commands need to be executed once:

??? note "initialization"
    === "CLI"
        ``` bash
        nerbb init
        nerbb download    # optional, if built-in datasets shall be used
        ```
    === "Python"
        ``` python
        nerbb.init()
        nerbb.download()  # optional, if built-in datasets shall be used
        ```

This creates a ``./data`` directory with the following structure:

``` xml
data/
└── datasets
    └── conll2003                     # built-in dataset, requires download
        └── train.csv
        └── val.csv
        └── test.csv
    └── [..]                          # more built-in datasets, requires download
└── experiment_configs
    └── default.ini                   # experiment config default values
    └── my_experiment.ini             # experiment config example
    └── my_experiment_conll2003.ini   # experiment config template
    └── [..]                          # more experiment config templates
└── pretrained_models                 # custom model checkpoints
└── results
```

-----------
## 2. Experiment

Fine-tuning a **specific model** on a **specific dataset** using **specific training (hyper)parameters** is called an **experiment**.

An experiment is defined by an **experiment configuration** file ``./data/experiment_configs/<experiment_name>.ini``.
One can view an experiment configuration as follows:

??? note "show experiment configuration"
    === "CLI"
        ``` bash
        nerbb show_experiment_config <experiment_name>
        ```
    === "Python"
        ``` python
        nerbb.show_experiment_config("<experiment_name>")
        ```

### a. Use a predefined experiment

For now, we use the predefined experiment configuration with ``<experiment_name> = my_experiment``.

### b. Run an experiment

Once an experiment is defined, the following command can be used to run it.

??? note "run experiment"
    === "CLI"
        ``` bash
        nerbb run_experiment <experiment_name>
        ```
    === "Python"
        ``` python
        nerbb.run_experiment("<experiment_name>")
        ```

### c. Get experiment results

Once an experiment is finished, one can inspect the main results or detailed results:

??? note "get main results"
    === "CLI"
        ``` bash
        nerbb get_experiment_results <experiment_name>  # prints overview on runs
        ```
    === "Python"
        ``` python
        experiment_results = nerbb.get_experiment_results("<experiment_name>")
        ```

    Python: see [ExperimentResults](../../python_api/experiment_results) for details on how to use ``experiment_results``

??? note "get detailed results & run histories using either mlflow or tensorboard"
  
    === "CLI"
        ``` bash
        nerbb mlflow       # + enter http://localhost:5000 in your browser
        nerbb tensorboard  # + enter http://localhost:6006 in your browser
        ```

### d. Predict tags using the best model

??? note "predict tags using the best model"
    === "CLI"
        ``` bash
        # e.g. <text_input> = "annotera den här texten"
        nerbb predict <experiment_name> <text_input>
        ```
    === "Python"
        ``` python
        # e.g. <text_input> = "annotera den här texten"
        nerbb.predict("<experiment_name>", <text_input>)

        # same but w/o having to reload the best model for multiple predictions
        experiment_results = nerbb.get_experiment_results(<experiment_name>)
        experiment_results.best_model.predict(<text_input>)
        ```

    Python: see [NerModelPredict](../../python_api/ner_model_predict) for details on how to use ``experiments_results.best_model``

-----------
## 3. Experiments Overview

Once one or more experiments have been run, the following commands can be used to access their results:

??? note "get experiments overview"
    === "CLI"
        ``` bash
        nerbb get_experiments
        ```
    === "Python"
        ``` python
        nerbb.get_experiments()
        ```

??? note "get overview on experiments' best runs"
    === "CLI"
        ``` bash
        nerbb get_experiments_results
        ```
    === "Python"
        ``` python
        experiments_results = nerbb.get_experiments_results()
        ```

    Python: see [ExperimentsResults](../../python_api/experiments_results) for details on how to use ``experiments_results``

-----------
## 4. Next Steps

- Create your own [`Custom Experiments`](../custom_experiments).
- See [`Datasets and Models`](../datasets_and_models) 
  for an overview on **built-in** datasets and models, and learn how to include your own **custom** datasets and models.


# Getting Started

Use either the [`Python API`](../../python_api/overview) or the [`CLI (Command Line Interface)`](../../cli/cli).

??? note "basic usage"
    === "Python"
        ``` python
        from nerblackbox import NerBlackBox
        nerbb = NerBlackBox()
        ```
    === "CLI"
        ``` bash
        nerbb --help
        ```

-----------
## 1. Initialization

The following commands need to be executed once:

??? note "initialization"
    === "Python"
        ``` python
        nerbb.init()
        nerbb.download()  # optional, if built-in datasets shall be used
        ```
    === "CLI"
        ``` bash
        nerbb init
        nerbb download    # optional, if built-in datasets shall be used
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
## 2. Data

- [`Built-in datasets`](../datasets_and_models/#built-in-datasets) do not require any preparation. 

- [`Custom datasets`](../datasets_and_models/#custom-datasets) can be provided using two different data formats:

    - jsonl (raw data)

    - csv (pretokenized data)

    See [`Custom datasets`](../datasets_and_models/#custom-datasets) for more details.

-----------
## 3. Fine-tune a Model

Fine-tuning a **specific model** on a **specific dataset** using **specific parameters** is called an **experiment**. 

An experiment is defined 

- either **statically** by an **experiment configuration** file ``./data/experiment_configs/<experiment_name>.ini``.

    ??? note "run experiment statically"
        === "Python"
            ``` python
            nerbb.run_experiment("<experiment_name>", from_config=True)
            ```
        === "CLI"
            ``` bash
            nerbb run_experiment <experiment_name>
            ```

- or **dynamically** using function arguments in [`nerbb.run_experiment()`](../../python_api/nerblackbox/#nerblackbox.api.NerBlackBox.run_experiment)

    ??? note "run experiment dynamically (only Python API)"
        === "Python"
            ``` python
            nerbb.run_experiment("<experiment_name>", model="<model_name>", dataset="<dataset_name>")
            ```

    This creates an experiment configuration on the fly, which is subsequently used.

In both cases, the specification of the [`model`](../datasets_and_models) and the [`dataset`](../datasets_and_models) are mandatory, while the [`parameters`](../parameters) are all optional. The hyperparameters that are used by default are globally applicable settings that should give close-to-optimal results for any use case.
In particular, [`adaptive fine-tuning`](../../features/adaptive_finetuning) is employed to ensure that this holds irrespective of the size of the dataset.  

-----------
## 4. Inspect the Model

- Once an experiment is finished, one can inspect its main results 
    or have a look at detailed results (e.g. learning curves):

    ??? note "get main results"
        === "Python"
            ``` python
            experiment_results = nerbb.get_experiment_results("<experiment_name>")  # List[ExperimentResults]
            ```
        === "CLI"
            ``` bash
            nerbb get_experiment_results <experiment_name>  # prints overview on runs
            ```

        Python: see [ExperimentResults](../../python_api/experiment_results) for details on how to use ``experiment_results``

    ??? note "get detailed results (only CLI)"
      
        === "CLI"
            ``` bash
            nerbb mlflow       # + enter http://localhost:5000 in your browser
            nerbb tensorboard  # + enter http://localhost:6006 in your browser
            ```

    See [Detailed Analysis of Training Results](../../features/detailed_results) for more information.

- An overview of all experiments and their results can be accessed as follows:

    ??? note "get overview of all experiments"
        === "Python"
            ``` python
            nerbb.get_experiments()
            ```
        === "CLI"
            ``` bash
            nerbb get_experiments
            ```

    ??? note "get overview of all experiments' main results"
        === "Python"
            ``` python
            experiment_results_all = nerbb.get_experiment_results("all")  # List[ExperimentResults]
            ```
        === "CLI"
            ``` bash
            nerbb get_experiment_results all
            ```

        Python: see [ExperimentResults](../../python_api/experiment_results) for details on how to use ``experiment_results_all``

-----------
## 5. Model Inference

??? note "model inference"
    === "Python"
        ``` python
        # e.g. <text_input> = "annotera den här texten"
        nerbb.predict("<experiment_name>", <text_input>)

        # same but w/o having to reload the best model for multiple predictions
        ner_model_predict = nerbb.get_model_from_experiment(<experiment_name>)
        ner_model_predict.predict(<text_input>)
        ```
    === "CLI"
        ``` bash
        # e.g. <text_input> = "annotera den här texten"
        nerbb predict <experiment_name> <text_input>
        ```

    Python: see [NerModelPredict](../../python_api/ner_model_predict) for further details on how to use ``ner_model_predict``

-----------
## Next Steps

- See [`Datasets and Models`](../datasets_and_models) to learn how to include your own **custom datasets** and **custom models**.
- See [`Parameters`](../parameters) for information on how to create your own **custom experiments**.


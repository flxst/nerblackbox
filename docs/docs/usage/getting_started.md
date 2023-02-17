# Getting Started

**nerblackbox** provides a [Python API](../../python_api/overview) with four main classes 
`Store`, `Dataset`, `Experiment` and `Model`.
Alternatively, a [CLI (Command Line Interface)](../../cli/cli) with the command `nerbb` is available.

??? note "basic usage"
    === "Python"
        ``` python
        from nerblackbox import Store, Dataset, Experiment, Model
        ```
    === "CLI"
        ``` bash
        nerbb --help
        ```

-----------
## 1. Store

First, a store has to be created. 
The store is a directory that contains all the data 
(datasets, experiment configurations, results, model checkpoints)
that **nerblackbox** needs access to. 
It is handled by the [Store](../../python_api/store) class:

??? note "create store"
    === "Python"
        ``` python
        Store.create()
        ```
    === "CLI"
        ``` bash
        nerbb create
        ```

By default, the store is located at ``./store`` and has the following subdirectories:

``` xml
store/
└── datasets
└── experiment_configs
└── pretrained_models
└── results
```

If wanted, the path of the store can be adjusted (before creation) like this:

??? note "adjust store path"
    === "Python"
        ``` python
        Store.set_path("<store_path>")
        ```
    === "CLI"
        ``` bash
        nerbb --store_dir <store_dir> create
        ```

-----------
## 2. Dataset

Next, a dataset needs to be prepared.

- [Built-in datasets](../datasets_and_models/#built-in-datasets) (including [HuggingFace Datasets](../../features/support_huggingface_datasets/))
can easily be downloaded and set up using the [Dataset](../../python_api/dataset) class:


    ??? note "set up a built-in dataset"
        === "Python"
            ``` python
            dataset = Dataset("<dataset_name>")
            dataset.set_up()
            ```
        === "CLI"
            ``` bash
            nerbb set_up_dataset <dataset_name>
            ```

    This creates dataset files in the folder `./store/datasets/<dataset_name>`.

- [Custom datasets](../datasets_and_models/#custom-datasets) can be added manually using two different data formats:

    - jsonl (raw data)

    - csv (pretokenized data)

    See [Custom datasets](../datasets_and_models/#custom-datasets) for more details.

-----------
## 3. Experiment (fine-tune a model)

Fine-tuning a **specific model** on a **specific dataset** using **specific parameters** is called an **experiment**. 
Everything around an experiment is handled by the [Experiment](../../python_api/experiment) class.

-----------
### 3a. Define an experiment

An experiment is defined 

- either **dynamically** through arguments when an [Experiment](../../python_api/experiment/) instance is created

    ??? note "define experiment dynamically (only Python API)"
        === "Python"
            ``` python
            experiment = Experiment("<experiment_name>", model="<model_name>", dataset="<dataset_name>")
            ```

- or **statically** by an **experiment configuration** file ``./store/experiment_configs/<experiment_name>.ini``.

    ??? note "define experiment statically"
        === "Python"
            ``` python
            experiment = Experiment("<experiment_name>", from_config=True)
            ```
        === "CLI"
            see 3b.

Note that the dynamic variant also creates an experiment configuration, which is subsequently used.
In both cases, the specification of the [`model`](../datasets_and_models) and the [`dataset`](../datasets_and_models) are mandatory, while the [`parameters`](../parameters_and_presets/#parameters) are all optional. The hyperparameters that are used by default are globally applicable settings that should give close-to-optimal results for any use case.
In particular, [adaptive fine-tuning](../../features/training/adaptive_finetuning) is employed to ensure that this holds irrespective of the size of the dataset.  

-----------
### 3b. Run an experiment

A fine-tuning experiment is run using the following command:

??? note "run experiment"
    === "Python"
        ``` python
        experiment.run()
        ```
    === "CLI"
        ``` bash
        nerbb run_experiment <experiment_name>  # CLI: only static experiment definition
        ```

See [`Experiment.run()`](../../python_api/experiment/#nerblackbox.api.experiment.Experiment.run) for further details.

-----------
### 3c. Get the results

When an experiment is finished, one can get its main results like so:

??? note "get main results"
    === "Python"
        ``` python
        experiment.get_result(metric="f1", level="entity", phase="test")
        ```

See [`Experiment.get_result()`](../../python_api/experiment/#nerblackbox.api.experiment.Experiment.get_result) for further details.

-----------
## 4. Model

-----------
### 4a. Inference

The best model of an experiment can be loaded and used for inference using the following commands:

??? note "model inference"
    === "Python"
        ``` python
        model = Model.from_experiment(<experiment_name>)
        model.predict(<text_input>)
        ```
    === "CLI"
        ``` bash
        nerbb predict <experiment_name> "<text_input>"
        ```
<!---
### 4b. Evaluation

Any model can easily be evaluated on any dataset:

??? note "model evaluation"
    === "Python"
        ``` python
        model = Model.from_checkpoint("<checkpoint_path>")
        model.evaluate("<dataset_name>")
        model.get_result(metric="f1", level="entity", phase="test")
        # 0.9234
        ```
--->

See [Model](../../python_api/model) for further details on how to use the ``model``

-----------
## 5. Store (advanced)

The [Store](../../python_api/store) class provides a few additional useful methods.

- An overview of all experiments and their results can be accessed as follows:

    ??? note "get overview of all experiments"
        === "Python"
            ``` python
            Store.show_experiments()
            ```
        === "CLI"
            ``` bash
            nerbb show_experiments
            ```

- Detailed experiment results (e.g. learning curves) can be accessed using `mlflow` or `tensorboard`:

    ??? note "get detailed results"

        === "Python"
            ``` python
            Store.mlflow("start")       # + enter http://localhost:5000 in your browser
            Store.tensorboard("start")  # + enter http://localhost:6006 in your browser
            ```
        === "CLI"
            ``` bash
            nerbb mlflow         # + enter http://localhost:5000 in your browser
            nerbb tensorboard    # + enter http://localhost:6006 in your browser
            ```

        Python: The underlying processes can be stopped using 
        [`Store.mlflow("stop")`](../../python_api/store/#nerblackbox.api.store.Store.mlflow) 
        and [`Store.tensorboard("stop")`](../../python_api/store/#nerblackbox.api.store.Store.tensorboard).

    See [Detailed Analysis of Training Results](../../features/training/detailed_results) for more information.


-----------
## Next Steps

- See [Datasets and Models](../datasets_and_models) to learn how to include your own **custom datasets** and **custom models**.
- See [Parameters and Presets](../parameters_and_presets) for information on how to create your own **custom experiments**.


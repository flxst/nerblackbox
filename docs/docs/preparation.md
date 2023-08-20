# Preparation

-----------
## Basic Usage

**nerblackbox** provides a [Python API](../python_api/overview) with four main classes 
`Store`, `Dataset`, `Training` and `Model`.
It is complemented by a [CLI](../../cli).

??? note "basic usage"
    === "Python"
        ``` python
        from nerblackbox import Store, Dataset, Training, Model
        ```
    === "CLI"
        ``` bash
        nerblackbox --help
        ```

-----------
## Store

As a mandatory first step, a store has to be created. 
The store is a directory that contains all the data 
(datasets, training configurations, results, model checkpoints)
that **nerblackbox** needs access to. 
It is handled by the [Store](../python_api/store) class:

??? note "create store"
    === "Python"
        ``` python
        Store.create()
        ```

By default, the store is located at ``./store`` and has the following subdirectories:

``` xml
store/
└── datasets
└── training_configs
└── pretrained_models
└── results
```

If wanted, the path of the store can be adjusted (before creation) like this:

??? note "adjust store path"
    === "Python"
        ``` python
        Store.set_path("<store_path>")
        ```

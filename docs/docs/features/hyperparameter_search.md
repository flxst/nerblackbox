# Hyperparameter Search

A hyperparameter grid search can easily be conducted as part of an experiment.
The hyperparameters one wants to vary are to be specified in special sections ``[runA]``, ``[runB]`` etc. in the experiment configuration file.

??? example "Example: custom_experiment.ini (Hyperparameter Search)"
    ``` markdown
    [runA]
    batch_size = 16
    max_seq_length = 128
    lr_max = 2e-5
    lr_schedule = constant

    [runB]
    batch_size = 32
    max_seq_length = 64
    lr_max = 3e-5
    lr_schedule = cosine
    ```

    This creates **2 hyperparameter runs** (`runA` & `runB`).

See [`Hyperparameters`](../../usage/parameters/#4-hyperparameters) for an overview of all hyperparameters.

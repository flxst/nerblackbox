# Parameters and Presets

An experiment is defined by a set of parameters. 

These can be specified in a [static experiment configuration](../getting_started/#3-experiment-fine-tune-a-model) file ``./store/experiment_configs/<experiment_name>.ini``.

??? example "Example: custom_experiment.ini"
    ``` markdown
    [dataset]
    dataset_name = swedish_ner_corpus
    annotation_scheme = plain
    prune_ratio_train = 0.1  # for testing
    prune_ratio_val = 1.0
    prune_ratio_test = 1.0
    train_on_val = False
    train_on_test = False

    [model]
    pretrained_model_name = af-ai-center/bert-base-swedish-uncased

    [settings]
    checkpoints = True
    logging_level = info
    multiple_runs = 1
    seed = 42

    [hparams]
    max_epochs = 250
    early_stopping = True
    monitor = val_loss
    min_delta = 0.0
    patience = 0
    mode = min
    lr_warmup_epochs = 2
    lr_num_cycles = 4
    lr_cooldown_restarts = True
    lr_cooldown_epochs = 7

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

Alternatively, the parameters can be used to [define an experiment dynamically](../getting_started/#3-experiment-fine-tune-a-model).
In that case, there are several hyperparameter [presets](./#presets) available 
for use with [Experiment()](../../python_api/experiment).

In the following, we will go through the different parameters step by step to see what they mean.


-----------
## Parameters

An experiment configuration contains the following **parameter groups**:

1. Dataset
2. Model
3. Settings
4. Hyperparameters

Some parameters are **mandatory** (i.e. they have to be included in an experiment configuration), 
others are **optional** and are set to default values if not specified.

-----------
### 1. Dataset

| Key               | Mandatory | Default Value | Type  | Values           | Comment                                                                                                                                                        |          
|---                |---        |---            |---    |---               |---                                                                                                                                                             |
| dataset_name      | Yes       | ---           | str   | e.g. conll2003   | [Built-in Dataset](../datasets_and_models/#built-in-datasets) or [Custom Dataset](../datasets_and_models/#custom-datasets)                                     |
| annotation_scheme | No        | auto          | str   | auto, plain, bio | specify dataset tag format (e.g. [BIO](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))). auto means it is inferred from data |
| prune_ratio_train | No        | 1.0           | float | 0.0 - 1.0        | fraction of train dataset to be used                                                                                                                           |
| prune_ratio_val   | No        | 1.0           | float | 0.0 - 1.0        | fraction of val   dataset to be used                                                                                                                           | 
| prune_ratio_test  | No        | 1.0           | float | 0.0 - 1.0        | fraction of test  dataset to be used                                                                                                                           |
| train_on_val      | No        | False         | bool  | True, False      | whether to train additionally on validation dataset                                                                                                           |
| train_on_test     | No        | False         | bool  | True, False      | whether to train additionally on test dataset                                                                                                           |

??? example "Example: custom_experiment.ini (Dataset)"
    ``` markdown
    [dataset]
    dataset_name = swedish_ner_corpus
    annotation_scheme = plain
    prune_ratio_train = 0.1  # for testing
    prune_ratio_val = 1.0
    prune_ratio_test = 1.0
    train_on_val = False
    train_on_test = False
    ```

-----------
### 2. Model

| Key                   | Mandatory | Default Value | Type  | Values                                      | Comment                                                                                                            |          
|---                    |---        |---            |---    |---                                          |---                                                                                                                 |
| pretrained_model_name | Yes       | ---           | str   | e.g. af-ai-center/bert-base-swedish-uncased | [Built-in Model](../datasets_and_models/#built-in-models) or [Custom Model](../datasets_and_models/#custom-models) |

??? example "Example: custom_experiment.ini (Model)"
    ``` markdown
    [model]
    pretrained_model_name = af-ai-center/bert-base-swedish-uncased
    ```
-----------
### 3. Settings

| Key               | Mandatory | Default Value | Type  | Values         | Comment                                                                                              |          
|---                |---        |---            |---    |---             |------------------------------------------------------------------------------------------------------|
| checkpoints       | No        | True          | bool  | True, False    | whether to save model checkpoints                                                                    |
| logging_level     | No        | info          | str   | info, debug    | choose [logging level](https://docs.python.org/3/library/logging.html#levels), debug is more verbose |
| multiple_runs     | No        | 1             | int   | 1+             | choose how often each hyperparameter run is executed (to control for statistical uncertainties)      |
| seed              | No        | 42            | int   | 1+             | for reproducibility. multiple runs get assigned different seeds.                                     |

??? example "Example: custom_experiment.ini (Settings)"
    ``` markdown
    [settings]
    checkpoints = True
    logging_level = info
    multiple_runs = 1
    seed = 42
    ```

-----------
### 4. Hyperparameters

| Key                  | Mandatory | Default Value | Type  | Values                   | Comment                                                                                                                |          
|---                   |---        |---------------|---    |---                       |---                                                                                                                     |
| batch_size           | No        | 16            | int   | e.g. 16, 32, 64          | number of training samples in one batch             |
| max_seq_length       | No        | 128           | int   | e.g. 64, 128, 256        | maximum sequence length used for model's input data |
| max_epochs           | No        | 250           | int   | 1+                       | (maximum) amount of training epochs                                                                                    |
| early_stopping       | No        | True          | bool  | True, False              | whether to use early stopping                                                                                          |
| monitor              | No        | val_loss      | str   | val_loss, val_acc        | if early stopping is True: metric to monitor (acc = accuracy)                                                                     |
| min_delta            | No        | 0.0           | float | 0.0+                     | if early stopping is True: minimum amount of improvement (w.r.t. monitored metric) required to continue training                  |
| patience             | No        | 0             | int   | 0+                       | if early stopping is True: number of epochs to wait for improvement w.r.t. monitored metric until training is stopped             |
| mode                 | No        | min           | str   | min, max                 | if early stopping is True: whether the optimum for the monitored metric is the minimum (val_loss) or maximum (val_acc) value      |
| lr_warmup_epochs     | No        | 2             | int   | 0+                       | number of epochs to linearly increase the learning rate during the warm-up phase, gets translated to [num_warmup_steps](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_scheduler) |
| lr_max               | No        | 2e-5          | float | e.g. 2e-5, 3e-5          | maximum learning rate (after warm-up) for [AdamW optimizer](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.AdamW)       |
| lr_schedule          | No        | constant      | str   | constant, linear, cosine, cosine_with_hard_restarts, hybrid | [Learning Rate Schedule](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#schedules), i.e. how to vary the learning rate (after warm-up). hybrid = constant + linear cool-down. |
| lr_num_cycles        | No        | 4             | int   | 1+                       | num_cycles for [lr_schedule = cosine](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_schedule_with_warmup) or [lr_schedule = cosine_with_hard_restarts](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_with_hard_restarts_schedule_with_warmup) |
| lr_cooldown_restarts | No        | True          | bool  | True, False              | if early stopping is True: whether to restart normal training if monitored metric improves during cool-down phase                          |
| lr_cooldown_epochs   | No        | 7             | int   | 0+                       | if early stopping is True or lr_schedule == hybrid: number of epochs to linearly decrease the learning rate during the cool-down phase                          |

??? example "Example: custom_experiment.ini (Hyperparameters)"
    ``` markdown
    [hparams]
    max_epochs = 250
    early_stopping = True
    monitor = val_loss
    min_delta = 0.0
    patience = 0
    mode = min
    lr_warmup_epochs = 2
    lr_num_cycles = 4
    lr_cooldown_restarts = True
    lr_cooldown_epochs = 7

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

    This creates **2 hyperparameter runs** (`runA` & `runB`). Each hyperparameter run is executed **multiple_runs** times (see [3. Settings](#3-settings)).

-----------
## Presets

When an experiment is defined [dynamically](../getting_started/#3-experiment-fine-tune-a-model), there are several hyperparameter presets available.
They can be specified using the ``from_preset`` argument in [Experiment()](../../python_api/experiment).

In the following, we list the different presets together with the [Hyperparameters](./#4-hyperparameters) that they entail:

- ``from_preset = adaptive``

    ??? note "adaptive fine-tuning hyperparameters"
        ``` markdown
        [hparams]
        max_epochs = 250
        early_stopping = True
        monitor = val_loss
        min_delta = 0.0
        patience = 0
        mode = min
        lr_warmup_epochs = 2
        lr_schedule = constant
        lr_cooldown_epochs = 7
        ```

- ``from_preset = original``

    ??? note "original fine-tuning hyperparameters"
        ``` markdown
        [hparams]
        max_epochs = 5
        early_stopping = False
        lr_warmup_epochs = 2
        lr_schedule = linear
        ```

- ``from_preset = stable``

    ??? note "stable fine-tuning hyperparameters"
        ``` markdown
        [hparams]
        max_epochs = 20
        early_stopping = False
        lr_warmup_epochs = 2
        lr_schedule = linear
        ```

More information on the different approaches behind the presets can be found [here]().
# Custom Experiments

An experiment is defined by an experiment configuration file ``./data/experiment_configs/<experiment_name>.ini``.

Create your own **custom experiment configuration** with ``<experiment_name> = custom_experiment``.

??? example "Example: custom_experiment.ini"
    ``` markdown
    [dataset]
    dataset_name = swedish_ner_corpus
    dataset_tags = iob
    prune_ratio_train = 0.1  # for testing
    prune_ratio_val = 1.0
    prune_ratio_test = 1.0

    [model]
    pretrained_model_name = af-ai-center/bert-base-swedish-uncased

    [settings]
    checkpoints = True
    logging_level = info
    multiple_runs = 3

    [hparams]
    max_epochs = 20
    monitor = val_loss
    min_delta = 0.0
    patience = 2
    mode = min
    lr_warmup_epochs = 1
    lr_num_cycles = 4

    [runA]
    batch_size = 16
    max_seq_length = 64
    lr_max = 2e-5
    lr_schedule = constant

    [runB]
    batch_size = 32
    max_seq_length = 128
    lr_max = 3e-5
    lr_schedule = cosine
    ```

In the following, we will go through the different parameters step by step to see what they mean.


-----------
## Parameters

An experiment configuration contains the following **parameter groups**:

1. Dataset
2. Model
3. Settings
4. Hyperparameters (Optional)
5. Hyperparameters (Mandatory)

Some parameters are **mandatory** (i.e. they have to be included in an experiment configuration), 
others are **optional** and are set to default values if not specified.

-----------
### 1. Dataset

| Key               | Mandatory | Default Value | Type  | Values         | Comment                                                                                                                                |          
|---                |---        |---            |---    |---             |---                                                                                                                                     |
| dataset_name      | Yes       | ---           | str   | e.g. conll2003 | [Built-in Dataset](../datasets_and_models/#built-in-datasets) or [Custom Dataset](../datasets_and_models/#custom-datasets)             |
| dataset_type      | Yes       | ---           | str   | iob, plain     | specify if dataset tags are in [IOB](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) or plain format |
| prune_ratio_train | No        | 1.0           | float | 0.0 - 1.0      | fraction of train dataset to be used                                                                                                   |
| prune_ratio_val   | No        | 1.0           | float | 0.0 - 1.0      | fraction of val   dataset to be used                                                                                                   | 
| prune_ratio_test  | No        | 1.0           | float | 0.0 - 1.0      | fraction of test  dataset to be used                                                                                                   |

??? example "Example: custom_experiment.ini (Dataset)"
    ``` markdown
    [dataset]
    dataset_name = swedish_ner_corpus
    dataset_tags = iob
    prune_ratio_train = 0.1  # for testing
    prune_ratio_val = 1.0
    prune_ratio_test = 1.0
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
|---                |---        |---            |---    |---             |---                                                                                                   |
| checkpoints       | No        | True          | bool  | True, False    | whether to save model checkpoints                                                                    |
| logging_level     | No        | info          | str   | info, debug    | choose [logging level](https://docs.python.org/3/library/logging.html#levels), debug is more verbose |
| multiple_runs     | No        | 1             | int   | 1+             | choose how often each hyperparameter run is executed (to control for statistical uncertainties)      |

??? example "Example: custom_experiment.ini (Settings)"
    ``` markdown
    [settings]
    checkpoints = True
    logging_level = info
    multiple_runs = 3
    ```

-----------
### 4. Hyperparameters (Optional)

| Key               | Mandatory | Default Value | Type  | Values                   | Comment                                                                                                                |          
|---                |---        |---            |---    |---                       |---                                                                                                                     |
| max_epochs        | No        | 20            | int   | 1+                       | maximum amount of training epochs                                                                                      |
| monitor           | No        | val_loss      | str   | val_loss, val_acc        | metric to monitor for early stopping (acc = accuracy)                                                                                  |
| min_delta         | No        | 0.0           | float | 0.0+                     | minimum amount of improvement (w.r.t. monitored metric) required to continue training (i.e. not employ early stopping) |
| patience          | No        | 2             | int   | 0+                       | number of epochs to wait for improvement w.r.t. monitored metric until early stopping is employed                      |
| mode              | No        | min           | str   | min, max                 | whether the optimum for the monitored metric is the minimum (val_loss) or maximum (val_acc) value                      |
| lr_warmup_epochs  | No        | 1             | int   | 0+                       | number of epochs to gradually increase the learning rate during the warm-up phase, gets translated to [num_warmup_steps](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_scheduler)                          |
| lr_num_cycles     | No        | 4             | int   | 1+                       | num_cycles for [lr_schedule = cosine](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_schedule_with_warmup) or [lr_schedule = cosine_with_hard_restarts](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_cosine_with_hard_restarts_schedule_with_warmup)                                                                                |

??? example "Example: custom_experiment.ini (Hyperparameters Optional)"
    ``` markdown
    [hparams]
    max_epochs = 20
    monitor = val_loss
    min_delta = 0.0
    patience = 2
    mode = min
    lr_warmup_epochs = 1
    lr_num_cycles = 4
    ```

-----------
### 5. Hyperparameters (Mandatory)

| Key               | Mandatory | Default Value | Type  | Values                                              | Comment                                             |          
|---                |---        |---            |---    |---                                                  |---                                                  |
| batch_size        | Yes       | ---           | int   | e.g. 16, 32, 64                                     | number of training samples in one batch             |
| max_seq_length    | Yes       | ---           | int   | e.g. 64, 128, 256                                   | maximum sequence length used for model's input data |
| lr_max            | Yes       | ---           | float | e.g. 2e-5, 3e-5                                     | maximum learning rate (after warm-up) for [AdamW optimizer](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.AdamW)                |
| lr_schedule       | Yes       | ---           | str   | constant, linear, cosine, cosine_with_hard_restarts | [Learning Rate Schedule](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#schedules), i.e. how to vary the learning rate (after warm-up)       |

???+ example "Example: custom_experiment.ini (Hyperparameters Mandatory)"
    ``` markdown
    [runA]
    batch_size = 16
    max_seq_length = 64
    lr_max = 2e-5
    lr_schedule = constant

    [runB]
    batch_size = 32
    max_seq_length = 128
    lr_max = 3e-5
    lr_schedule = cosine
    ```

    This creates **2 hyperparameter runs** (`runA` & `runB`). Each hyperparameter run is executed **multiple_runs** times (see [3. Settings](#3-settings)).

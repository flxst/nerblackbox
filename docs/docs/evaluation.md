# Evaluation

The [Model](../../../python_api/model) class provides the functionality to evaluate **any** NER model on **any** NER dataset.
Both the fine-tuned NER model and the dataset can either be loaded from [HuggingFace (HF)](https://huggingface.co) or the Local Filesystem (LF).

----------
## Usage

1) load a [Model](../../../python_api/model) instance:

??? note "load model"
    ``` python
    # from local checkpoint directory
    model = Model.from_checkpoint("<checkpoint_directory>")

    # from experiment
    model = Model.from_experiment("<experiment_name>")

    # from HuggingFace
    model = Model.from_huggingface("<repo_id>")
    ```

2) use the [evaluate_on_dataset()](../../../python_api/model/#nerblackbox.api.model.Model.evaluate_on_dataset) method:

??? note "model evaluation on dataset"
    ``` python
    # local dataset in standard format (jsonl)
    evaluation_dict = model.evaluate_on_dataset("<local_dataset_in_standard_format>", "jsonl", phase="test")

    # local dataset in pretokenized format (csv)
    evaluation_dict = model.evaluate_on_dataset("<local_dataset_in_pretokenized_format>", "csv", phase="test")

    # huggingface dataset in pretokenized format
    evaluation_dict = model.evaluate_on_dataset("<huggingface_dataset_in_pretokenized_format>", "huggingface", phase="test")
    ```

### Interpretation

The returned object `evaluation_dict` is a nested dictionary `evaluation_dict[label][level][metric]` where

- `label` in `['micro', 'macro']`
- `level` in `['entity', 'token']`
- `metric` in `['precision', 'recall', 'f1', 'precision_seqeval', 'recall_seqeval', 'f1_seqeval']`

??? example "evaluation_dict"
    ``` python
    evaluation_dict["micro"]["entity"]
    # {
    #   'precision': 0.912,
    #   'recall': 0.919,
    #   'f1': 0.916,
    #   'precision_seqeval': 0.907,
    #   'recall_seqeval': 0.919,
    #   'f1_seqeval': 0.913}
    # }
    ```

The metrics `precision`, `recall` and `f1` are **nerblackbox**'s evaluation results, whereas their counterparts with a `_seqeval` suffix correspond to the results you would get using the [**seqeval**](https://github.com/chakki-works/seqeval) library (which is also used by and [HuggingFace evaluate](https://huggingface.co/docs/evaluate/index)).
The difference lies in the way model predictions which are inconsistent with the employed [annotation scheme](../../data/support_annotation_schemes) are handled.
While **nerblackbox**'s evaluation ignores them, **seqeval** takes them into account.

### Example
A complete example of an evaluation using both the model and the dataset from HuggingFace:

??? example "complete evaluation example"
    ``` python
    # 1. load the model
    model = Model.from_huggingface("dslim/bert-base-NER")

    # 2. evaluate the model on the dataset
    evaluation_dict = model.evaluate_on_dataset("conll2003", "huggingface", phase="test")

    # 3. inspect the results
    evaluation_dict["micro"]["entity"]
    # {
    #   'precision': 0.912,
    #   'recall': 0.919,
    #   'f1': 0.916,
    #   'precision_seqeval': 0.907,
    #   'recall_seqeval': 0.919,
    #   'f1_seqeval': 0.913}
    # }
    ```
    Note that the **seqeval** results are in accordance with the [official results](https://huggingface.co/dslim/bert-base-NER).
    The **nerblackbox** results have a slightly higher precision (and f1 score).

# [Evaluation] Evaluation of a Model on a Dataset

The [Model](../../../python_api/model) class provides the functionalities to evaluate a model on a dataset.
Both the model and the dataset can either be loaded from local files or directly from the [HuggingFace Hub](https://huggingface.co/datasets).


----------
1) load the [Model](../../../python_api/model) instance:

??? note "from local checkpoint directory"
    ``` python
    model = Model.from_checkpoint("<checkpoint_directory>")
    ```

??? note "from local experiment"
    ``` python
    model = Model.from_experiment("<experiment_name>")
    ```
   
??? note "from huggingface"
    ``` python
    model = Model.from_huggingface("<repo_id>")
    ```

----------
2) use the [evaluate_on_dataset()](../../../python_api/model/#nerblackbox.api.model.Model.evaluate_on_dataset) method:

??? note "on local dataset in [standard format](../../data/support_pretokenized/)"
    ``` python
    evaluation_dict = model.evaluate_on_dataset("<local_dataset_in_standard_format>", "jsonl", phase="test")
    ```

??? note "on local dataset in [pretokenized format](../../data/support_pretokenized/)"
    ``` python
    evaluation_dict = model.evaluate_on_dataset("<local_dataset_in_pretokenized_format>", "csv", phase="test")
    ```

??? note "on huggingface dataset in [pretokenized format](../../data/support_pretokenized/)"
    ``` python
    evaluation_dict = model.evaluate_on_dataset("<huggingface_dataset_in_pretokenized_format>", "huggingface", phase="test")
    ```

----------
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

----------
A complete example of an evaluation using both the model and the dataset from the HuggingFace Hub:

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

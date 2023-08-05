# Inference

The [Model](../python_api/model) class provides the functionality for versatile model predictions on one or multiple documents.
The corresponding fine-tuned NER model can either be loaded from [HuggingFace (HF)](https://huggingface.co/models) or the Local Filesystem (LF).

-----------
## Basic Usage

1) load a [Model](../python_api/model) instance:

??? note "load model"
    === "Python"
        ``` python
        # from local checkpoint directory
        model = Model.from_checkpoint("<checkpoint_directory>")

        # from experiment
        model = Model.from_experiment("<experiment_name>")

        # from HuggingFace
        model = Model.from_huggingface("<repo_id>")
        ```

2) use the [predict()](../python_api/model/#nerblackbox.api.model.Model.predict) method:

??? note "model inference"
    === "Python"
        ``` python
        model.predict(<text_input>)
        ```

### Example

??? example "Basic Inference"
    === "Python"
        ``` python
        model = Model.from_experiment("my_experiment")

        # predict on entity level
        model.predict("The United Nations has never recognised Jakarta's move.", level="entity")  
        # [[
        #  {'char_start': '4', 'char_end': '18', 'token': 'United Nations', 'tag': 'ORG'},
        #  {'char_start': '40', 'char_end': '47', 'token': 'Jakarta', 'tag': 'LOC'}
        # ]]
        ```

-----------
## Advanced Usage

The [predict()](../python_api/model/#nerblackbox.api.model.Model.predict) method used in the example above might be the most important tool for inference. 
However, the [Model](../python_api/model) class provides multiple methods to cover different use cases:

- [predict_on_file()](../python_api/model/#nerblackbox.api.model.Model.predict_on_file) takes a `jsonl` file as input and writes another `jsonl` file with predictions on the entity level.
  This is for instance useful if one wants to annotate (large) amounts of text in production.
- [predict()](../python_api/model/#nerblackbox.api.model.Model.predict) takes a single `string` or a `list of strings` as input. It allows to inspect the model predictions on the entity or word level. This is useful for instance for development and debugging.
- [predict_proba()](../python_api/model/#nerblackbox.api.model.Model.predict_proba) is similar to predict(), but returns predictions on the word level only, together with their probabilities. This can be useful for instance in conjunction with active learning.

An overview is given in the following table:

| method                                                                                   | input                     | level         | probabilities |
|:-----------------------------------------------------------------------------------------|:--------------------------|:--------------|:--------------|
| [predict_on_file()](../python_api/model/#nerblackbox.api.model.Model.predict_on_file) | jsonl file with documents | entity | no
| [predict()](../python_api/model/#nerblackbox.api.model.Model.predict)                 | one or multiple documents | entity, word  | no            |
| [predict_proba()](../python_api/model/#nerblackbox.api.model.Model.predict_proba)                                               | one or multiple documents | word | yes |


### Example

??? example "Advanced Inference"
    === "Python"
        ``` python
        model = Model.from_experiment("my_experiment")

        # predict on entity level using file 
        model.predict_on_file("<input_file>", "<output_file>")  

        # predict on word level 
        model.predict("The United Nations has never recognised Jakarta's move.", level="word")  
        # [[
        #  {'char_start': '4', 'char_end': '18', 'token': 'United Nations', 'tag': 'ORG'},
        #  {'char_start': '40', 'char_end': '47', 'token': 'Jakarta', 'tag': 'LOC'}
        # ]]

        # predict probabilities on word level 
        model.predict_proba(["arbetsförmedlingen finns i stockholm"])
        # [[
        #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "proba_dist: {"O": 0.21, "B-ORG": 0.56, ..}},
        #     {"char_start": "19", "char_end": "24", "token": "finns", "proba_dist: {"O": 0.87, "B-ORG": 0.02, ..}},
        #     {"char_start": "25", "char_end": "26", "token": "i", "proba_dist: {"O": 0.95, "B-ORG": 0.01, ..}},
        #     {"char_start": "27", "char_end": "36", "token": "stockholm", "proba_dist: {"O": 0.14, "B-ORG": 0.22, ..}},
        # ]]
        ```

See [Model](../python_api/model) for further details.

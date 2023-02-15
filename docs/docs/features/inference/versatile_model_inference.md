# [Inference] Versatile Model Inference

The [Model](../../../python_api/model) class provides three methods for versative model inference:


| method                                                                                   | input                     | level         | probabilities |
|:-----------------------------------------------------------------------------------------|:--------------------------|:--------------|:--------------|
| [predict_on_file()](../../../python_api/model/#nerblackbox.api.model.Model.predict_on_file) | jsonl file with documents | entity | no
| [predict()](../../../python_api/model/#nerblackbox.api.model.Model.predict)                 | one or multiple documents | entity, word  | no            |
| [predict_proba()](../../../python_api/model/#nerblackbox.api.model.Model.predict_proba)                                               | one or multiple documents | word | yes |

Note that:

- [predict_on_file()](../../../python_api/model/#nerblackbox.api.model.Model.predict_on_file) takes a `jsonl` file as input and writes another `jsonl` file with predictions on the entity level. 
This is for instance useful if one wants to annotate (large) amounts of text in production.
- [predict()](../../../python_api/model/#nerblackbox.api.model.Model.predict) takes a single `string` or a `list of strings` as input. It allows to inspect the model predictions on the entity or word level. This is useful for instance for development and debugging.
- [predict_proba()](../../../python_api/model/#nerblackbox.api.model.Model.predict_proba) is similar to predict(), but returns predictions on the word level only, together with their probabilities. This can be useful for instance in conjunction with active learning.
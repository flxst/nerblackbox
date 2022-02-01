# Special Token Support

Transformer-based models handle out-of-vocabulary tokens by replacing them by an 
``[UNK]`` token. Sometimes, text contains special characters or tokens that contain information 
that is valuable for the model. 
For example, line breaks (``\n``) may occur to conclude a paragraph, 
and tabs (``\t``) may appear in the context of an enumeration.

**nerblackbox** allows to define a list of special tokens that are added to the model's vocabulary
in order to preserve the semantic information these contain.

Just add a `special_tokens.jsonl` file to the same folder ``./data/datasets/<custom_dataset>`` that contains the data, 
see [`Custom Datasets`](../../usage/datasets_and_models/#custom-datasets).


??? example "special_tokens.jsonl"
    ``` markdown
    ["[NEWLINE]", "[TAB]"]
    ```
    In this example, we assume that as part of the data preprocessing process, line breaks `\n` and tabs `\t` were replaced by the special tokens 
    `[NEWLINE]` and `[TAB]`, respectively. Those special tokens are then included
    in the `special_tokens.jsonl` file such that they get taken into account in the model training.



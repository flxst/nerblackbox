# [Data] Text Encoding

Text may contain whitespace characters (e.g. "\n", "\t") or special characters (e.g. "•", emojis) that a pre-trained model has never seen before. 
While the whitespace characters are ignored in the tokenization process, the special characters lead to out-of-vocabulary tokens which get replaced by 
``[UNK]`` tokens before being sent to the model. 
Sometimes, however, the ignored or replaced tokens contain semantic information 
that is valuable for the model and thus should be preserved.

Therefore, **nerblackbox** allows to customly map selected special characters to self-defined special tokens ("encoding"). 
The encoded text may then be used during training and inference.

Say we want to have the following replacements:

??? example "encoding"
    ``` python
    # map special characters to special tokens
    encoding = {
        '\n': '[NEWLINE]',
        '\t': '[TAB]',
        '•': '[DOT]',
    }
    ```

--------
The first step is to save the `encoding` in an `encoding.json` file
which is located in the same folder ``./store/datasets/<custom_dataset>`` that contains the data 
(see [Custom Datasets](../../usage/datasets_and_models/#custom-datasets)).

??? note "create encoding.json"
    ``` python
    import json

    with open('./store/datasets/<custom_dataset>/encoding.json', 'w') as file:
        json.dump(encoding, file)
    ```

This way, the special tokens are automatically added to the model's vocabulary during training.


--------
The second step is to apply the `encoding` to the data. 
The [TextEncoder](../../python_api/text_encoder) class 
takes care of this:

??? note "TextEncoder"
    ``` python
    from nerblackbox import TextEncoder

    text_encoder = TextEncoder(encoding)
    ```

For **training**, one needs to encode the input text like so:

??? note "text encoding (training)"
    ``` python
    # ..load input_text 

    # ENCODE
    # e.g. input_text             = 'We\n are in • Stockholm' 
    #      input_text_encoded     = 'We[NEWLINE] are in [DOT] Stockholm'
    input_text_encoded, _ = text_encoder.encode(input_text)  

    # ..save input_text_encoded and use it for training
    ```

For **inference**, the predictions also need to be mapped back to the original text, like so:

??? note "text encoding (inference)"
    ``` python
    # ENCODE
    # e.g. input_text             = 'We\n are in • Stockholm'
    #      input_text_encoded     = 'We[NEWLINE] are in [DOT] Stockholm'
    #      encode_decode_mappings = [(2, "\n", "[NEWLINE]"), (13, "•", "[DOT]")]
    input_text_encoded, encode_decode_mappings = text_encoder.encode(input_text)


    # PREDICT
    # e.g. predictions_encoded    = {'char_start': 25, 'char_end': 34, 'token': 'Stockholm', 'tag': 'LOC'}
    predictions_encoded = model.predict(input_text_encoded, level="entity")


    # DECODE
    # e.g. input_text_decoded     = 'We\n are in • Stockholm' 
    #      predictions            = {'char_start': 13, 'char_end': 22, 'token': 'Stockholm', 'tag': 'LOC'}
    input_text_decoded, predictions = text_encoder.decode(input_text_encoded,
                                                          encode_decode_mappings,
                                                          predictions_encoded)
    ```

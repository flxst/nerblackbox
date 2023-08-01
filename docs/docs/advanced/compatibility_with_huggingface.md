# Compatibility with HuggingFace

**nerblackbox** is heavily based on [HuggingFace Transformers](https://huggingface.co/transformers/). 
Moreover, [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) and [HuggingFace Evaluate](https://huggingface.co/docs/evaluate/index) are well-integrated, 
see [Integration of HuggingFace Datasets](../../data/support_huggingface_datasets) and [Evaluation of a Model on a Dataset](../../evaluation/support_evaluation), respectively.

Therefore, compatibility with HuggingFace is given.
In particular, 

- **nerblackbox**'s model checkpoints (and tokenizer files)  are identical to the ones from HuggingFace.

    ??? example "model checkpoint directory"
        === "Bash"
            ``` bash
            ls <checkpoint_directory>
            # config.json             
            # pytorch_model.bin
            # special_tokens_map.json 
            # tokenizer.json          
            # tokenizer_config.json   
            # vocab.txt
            ```

- After a [Model](../../../python_api/model) instance is created from a checkpoint, it contains a HuggingFace model and tokenizer as attributes:

    ??? example "model attributes"
        === "Python"
            ``` python
            model = Model(<checkpoint_directory>)
            
            print(type(model.model))
            # <class 'transformers.models.bert.modeling_bert.BertForTokenClassification'>

            print(type(model.tokenizer))
            # <class 'transformers.models.bert.tokenization_bert_fast.BertTokenizerFast'>
            ```

    Hence, `model.model` and `model.tokenizer` can be used like any other **transformers** model and tokenizer, respectively.



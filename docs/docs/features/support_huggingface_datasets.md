# Seamless Use of Datasets from HuggingFace Datasets

In addition to [built-in](../../usage/datasets_and_models/#built-in-datasets) and your own [custom](../../usage/datasets_and_models/#custom-datasets) datasets,
any dataset from [HuggingFace Datasets](https://huggingface.co/datasets) that is suitable for named entity recognition
can be used without further ado.

- static experiment definition

    ??? example "experiment_huggingface_dataset.ini"
        ``` markdown
        [dataset]
        dataset_name = ehealth_kd

        [model]
        pretrained_model_name = mrm8488/electricidad-base-discriminator
        ```

    ??? example "run experiment with huggingface dataset (statically)"
        ``` python
        nerbb.run_experiment("experiment_huggingface_dataset", from_config=True)
        ```

- dynamic experiment definition

    ??? example "run experiment with huggingface dataset (dynamically)"
        ``` python
        nerbb.run_experiment("experiment_huggingface_dataset", model="mrm8488/electricidad-base-discriminator", dataset="ehealth_kd")
        ```

**nerblackbox** automatically determines whether the employed dataset 
uses the [standard or pretokenized](../support_pretokenized/) format.

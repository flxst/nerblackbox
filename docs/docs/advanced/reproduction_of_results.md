# Reproduction of Results

Results reported on HuggingFace are reproduced with **nerblackbox** as a cross-check. 
We do this separately for:

- [Training](#training) (pretrained models are fine-tuned for NER and subsequently evaluated on the validation set)
- [Evaluation](#evaluation) (NER models are evaluated on either the validation or test set)

Note that all numbers refer to the micro-averaged f1 score as computed by the `seqeval` library, see [Evaluation](../../evaluation).

-----------
## Training

| Model                              | Dataset      | Parameters                                                                                                                | version  | nerblackbox  | reported    |                                                                                                         |                                      
|------------------------------------|--------------|---------------------------------------------------------------------------------------------------------------------------|----------|--------------|-------------|--------------------------------------------------------------------------------------------------------|
| bert-base-cased                    | conll2003    | {"from_preset": "original", "multiple_runs": 3, "max_epochs": 5, "lr_warmup_epochs": 0}                                   | `0.0.15` | 0.946(1)     | 0.951       | [reference](https://huggingface.co/dslim/bert-base-NER)                                                |
| distilbert-base-multilingual-cased | conll2003    | {"from_preset": "original", "multiple_runs": 3, "max_epochs": 3, "lr_warmup_epochs": 0}                                   | `0.0.15` | 0.943(1)     | 0.941       | [reference](https://huggingface.co/gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner) |
| dmis-lab/biobert-base-cased-v1.2   | ncbi_disease | {"from_preset": "original", "multiple_runs": 3, "max_epochs": 3, "lr_warmup_epochs": 0, "batch_size": 4}                  | `0.0.15` | 0.855(4)     | 0.845       | [reference](https://huggingface.co/jordyvl/biobert-base-cased-v1.2_ncbi_disease-softmax-labelall-ner)  |
| distilroberta-base                 | conll2003    | {"from_preset": "original", "multiple_runs": 3, "max_epochs": 6, "lr_warmup_epochs": 0, "batch_size": 32, "lr_max": 5e-5} | `0.0.15` | **0.953(1)** | **0.953**   | [reference](https://huggingface.co/philschmid/distilroberta-base-ner-conll2003)                        |                 
| microsoft/deberta-base             | conll2003    | {"from_preset": "original", "multiple_runs": 3, "max_epochs": 5, "lr_warmup_epochs": 0, "batch_size": 16, "lr_max": 5e-5} | `0.0.15` | 0.957(1)     | 0.961       | [reference](https://huggingface.co/geckos/deberta-base-fine-tuned-ner)                                 |               


Each model was fine-tuned multiple times using **nerblackbox** and different random seeds. 
The resulting uncertainty of the mean value is specified in parentheses and refers to the last digit. 
In all cases, the evaluation was conducted on the respective validation dataset.
Note that small, systematic differences are to be expected as the reported results were created using very similar 
yet slightly different hyperparameters.

The results may be reproduced using the following code:
??? note "reproduction of training results"
    === "Python"
        ``` python
        from nerblackbox import Dataset, Experiment

        dataset = Dataset(name=<dataset>, source="HF")
        dataset.set_up()

        parameters = {"from_preset": "original", [..]}
        experiment = Experiment("exp", model="<model>", dataset="<dataset>", **parameters)
        experiment.run()

        result = experiment.get_result(metric="f1", level="entity", phase="validation")
        print(result)
        ```
        Note that the first use of **nerblackbox** requires the creation of a [Store](../../preparation/#store).

-------------
## Evaluation

| Model                                                             | Dataset                     | Phase       | version   | nerblackbox | evaluate  | reported  |                                                                                                       |                                      
|-------------------------------------------------------------------|-----------------------------|-------------|-----------|-------------|-----------|-----------|-------------------------------------------------------------------------------------------------------|
| dslim/bert-base-NER                                               | conll2003                   | validation  | `0.0.15`  | **0.951**   | **0.951**   | **0.951** | [reference](https://huggingface.co/dslim/bert-base-NER)                                               |
| jordyvl/biobert-base-cased-v1.2_ncbi_disease-softmax-labelall-ner | ncbi_disease                | validation  | `0.0.15`  | **0.845**   | **0.845** | **0.845** | [reference](https://huggingface.co/jordyvl/biobert-base-cased-v1.2_ncbi_disease-softmax-labelall-ner) |
| fhswf/bert_de_ner                                                 | germeval_14                 | test        | `0.0.15`  | **0.818**   | **0.818**   | 0.829     | [reference](https://huggingface.co/fhswf/bert_de_ner)                                                              |
| philschmid/distilroberta-base-ner-conll2003                       | conll2003                   | validation  | `0.0.15`  | **0.955**   | **0.955**   | 0.953     | [reference](https://huggingface.co/philschmid/distilroberta-base-ner-conll2003)                                               |
| philschmid/distilroberta-base-ner-conll2003                       | conll2003                   | test        | `0.0.15`  | **0.913**   | **0.913**   | 0.907     | [reference](https://huggingface.co/philschmid/distilroberta-base-ner-conll2003)                                                                                                      |
| projecte-aina/roberta-base-ca-cased-ner                           | projecte-aina/ancora-ca-ner | test        | `0.0.15`  | **0.896**   | **0.896**   | 0.881     | [reference](https://huggingface.co/projecte-aina/roberta-base-ca-cased-ner)                                                                                                      |
| gagan3012/bert-tiny-finetuned-ner                                 | conll2003                   | validation  | `0.0.15`  | **0.847**   | ---         | 0.818     | [reference](https://huggingface.co/gagan3012/bert-tiny-finetuned-ner)                                              |
| malduwais/distilbert-base-uncased-finetuned-ner                   | conll2003                   | test        | `0.0.15`  | **0.894**   | ---         | 0.930     | [reference](https://huggingface.co/malduwais/distilbert-base-uncased-finetuned-ner)                                               |
| IIC/bert-base-spanish-wwm-cased-ehealth_kd                        | ehealth_kd                  | test        | `0.0.15`  | **0.825**   | ---         | 0.843     | [reference](https://huggingface.co/IIC/bert-base-spanish-wwm-cased-ehealth_kd)                                               |
| drAbreu/bioBERT-NER-BC2GM_corpus                                  | bc2gm_corpus                | test        | `0.0.15`  | **0.808**   | ---         | 0.815     | [reference](https://huggingface.co/drAbreu/bioBERT-NER-BC2GM_corpus)                                               |
| geckos/deberta-base-fine-tuned-ner                                | conll2003                   | validation  | `0.0.15`  | **0.962**   | ---         | 0.961     | [reference](https://huggingface.co/geckos/deberta-base-fine-tuned-ner)                                               |




The results may be reproduced using the following code:
??? note "reproduction of evaluation results"
    === "Python"
        ``` python
        ###############
        # nerblackbox
        ###############
        from nerblackbox import Model
        model = Model.from_huggingface("<model>", "<dataset>")
        results_nerblackbox = model.evaluate_on_dataset("<dataset>", phase="<phase>")
        print(results_nerblackbox["micro"]["entity"]["f1_seqeval"])

        ###############
        # evaluate
        ###############
        from datasets import load_dataset
        from evaluate import evaluator
        evaluator = evaluator("token-classification")
        data = load_dataset("<dataset>", split="<phase>")
        results_evaluate = evaluator.compute(
            model_or_pipeline="<model>",
            data=data,
            metric="seqeval",
        )
        print(results_evaluate["overall_f1"])
        ```
        Note that the first use of **nerblackbox** requires the creation of a [Store](../../preparation/#store).

Evaluation using the [evaluate](https://huggingface.co/docs/evaluate/index) library fails for some of the tested datasets (---). 
For all cases where it works, we find that results from **nerblackbox** and [evaluate](https://huggingface.co/docs/evaluate/index) are in agreement.
In contrast, the self-reported results on HuggingFace sometimes differ. 

<!---
Datasets that the evaluation did not work for:

- lener_br
---->


# Changelog
This project CURRENTLY DOES NOT adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.0.14 (2023-02-14)
#### Added
- Model: prediction on file
- Model: evaluation of any model on any dataset

#### Changed
- API: complete renewal using classes Store, Dataset, Experiment, Model
- Supported python versions: 3.8 to 3.11
- Dataset: no shuffling by default

#### Fixed
- Model: base model with NER classification head can be loaded

## 0.0.13 (2022-03-18)
#### Added
- NerModelPredict: GPU batch inference
- TextEncoder class for custom data preprocessing
- HuggingFace datasets integration: enable subsets
- HuggingFace datasets: support for sucx_ner

#### Changed
- NerModelPredict: improved inference time and data post-processing
- API: load best model of experiment directly (instead of via ExperimentResults)
- upgrade pytorch-lightning

## 0.0.12 (2022-02-14)
#### Added
- Adaptive fine-tuning
- Integration of HuggingFace Datasets
- Integration of raw (unpretokenized) data
- Integration of different annotation schemes and seamless conversion between them
- Option to specify experiments dynamically (instead of using a config file)
- Option to add special tokens
- New built-in dataset: Swe-NERC
- Use seeds for reproducibility

#### Changed
- Validation only on single metric (e.g. loss) during training
- Shuffling of all datasets (train, val, test)
- Results: epochs start counting from 1 instead of 0
- Results: compute standard version of macro-average, plus number of contributing classes
- Results: add precision and recall

#### Fixed
- All models that are based on WordPiece tokenizer work
- Early stopping: use last model instead of stopped epoch model

## 0.0.11 (2021-07-08)
#### Added
- NerModelPredict: predict on token or entity level
- Evaluation entity level: compute metrics for single labels
- Evaluation token level: confusion matrix
- Evaluation token & entity level: number of predicted classes

#### Changed
- Evaluation token level: use plain annotation scheme
- Migrate to pytorch-lightning==1.3.7, seqeval==1.2.2, mlflow==1.8.0


## 0.0.10 (2021-06-18)
#### Added
- special [NEWLINE] token can be used in input data
- CLI command "predict_proba"

#### Changed
- long input samples are automatically sliced before sent to model
- NerModelPredict: unknown tokens are restored (in external mode) 

#### Fixed
- NerModelPredict for local pretrained models
- NerModelEvaluation


## 0.0.9 (2021-04-05)
#### Added
- Swedish datasets (SIC & SUC 3.0)
- Python 3.9 support

#### Fixed
- CLI command "get_experiments_results"


## 0.0.8 (2021-01-24)
#### Added
- CLI command "nerbb download" (and corresponding python method) to download built-in datasets

#### Changed
- CLI command "nerbb init" (and corresponding python method) no longer automatically downloads built-in datasets


## 0.0.7 (2020-12-07)
#### Added
- NerModelPredict: option to predict probabilities instead of tags

#### Changed
- Exposure of main python classes at top level of package
- Renamed LightningNerModelPredict -> NerModelPredict
- Renamed LightningNerModelTrain -> NerModelTrain

#### Fixed
- use of local pretrained models
- loading of NerModelPredict from checkpoint


## 0.0.6 (2020-10-24)
#### Added
- New CLI command "nerbb clear_data" to clear checkpoints and results

#### Changed
- Dependencies cleaned up and simplified
- Experiment configuration file "exp_test.ini" improved

#### Fixed
- Boolean CLI options "--verbose" and "--fp16"


## 0.0.5 (2020-07-03)
First open-sourced version.

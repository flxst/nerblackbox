# Changelog
This project CURRENTLY DOES NOT adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

# Multiple Runs with Different Random Seeds

The results of a fine-tuning run depend on the employed random seed, see e.g. [this paper](https://arxiv.org/abs/2202.02617) for a discussion. 
One may conduct multiple runs with different seeds that are otherwise identical, in order to 

- get control over the uncertainties (see [Detailed Analysis of Training Results](../detailed_results/))

- get an improved model performance

Multiple runs can easily be specified in the experiment configuration.

??? example "Example: custom_experiment.ini (Settings / Multiple Runs)"
    ``` markdown
    [settings]
    multiple_runs = 3
    seed = 42
    ```
    
    This creates 3 runs with seeds 43, 44 and 45.


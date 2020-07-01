Fine-tuning can be done in a few simple steps using an "experiment configuration file"

.. code-block:: python

   # cat <experiment_name>.ini
   dataset_name = swedish_ner_corpus
   pretrained_model_name = af-ai-center/bert-base-swedish-uncased

and either the Command Line Interface (CLI) or the Python API:

.. code-block:: python

   # CLI
   nerbb run_experiment <experiment_name>          # fine-tune
   nerbb get_experiment_results <experiment_name>  # get results/performance
   nerbb predict <experiment_name> <text_input>    # apply best model

   # Python API
   nerbb = NerBlackBox()
   nerbb.run_experiment(<experiment_name>)         # fine-tune
   nerbb.get_experiment_results(<experiment_name>) # get results/performance
   nerbb.predict(<experiment_name>, <text_input>)  # apply best model

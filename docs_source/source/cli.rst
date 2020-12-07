
.. _cli:

CLI
===

Usage:

    .. code-block:: python

        nerbb [OPTIONS] FLAG [FLAG_ARGS]

Examples:

    .. code-block:: python

            nerbb analyze_data my_dataset_name
            nerbb --data_dir my_data_directory run_experiment my_experiment_name
            nerbb mlflow

.. click:: nerblackbox.cli:main
   :prog: nerbb
   :nested: full
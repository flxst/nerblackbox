# Notebooks

## Overview

| Title                          | Description                                                 | File              | Config file     |
|--------------------------------|-------------------------------------------------------------|-------------------|-----------------|
| NER with Data from HuggingFace | How to download data from HuggingFace and train a NER model | huggingface.ipynb | ---             |
| NER with Data from LabelStudio | How to annotate data with LabelStudio and train a NER model | labelstudio.ipynb | labelstudio.ini |
| NER with Data from Doccano     | How to annotate data with Doccano and train a NER model     | doccano.ipynb     | doccano.ini     |

## How to run the notebooks locally

It is recommended to run the notebooks in a virtual environment.

1. Create a virtual environment
   
    ```
    python3 -m venv venv-nerblackbox
    ```
   
2. Activate the virtual environment
    ```
    source venv-nerblackbox/bin/activate
    ```

3. Install nerblackbox
    ```
    pip install nerblackbox
    ```

4. Make the virtual environment accessible in notebooks:

    ```
    ipython kernel install --user --name=venv-nerblackbox
    ```
   
5. Run the notebook server
    ```
    jupyter notebook
    ```
   
6. Access the Jupyter UI in your browser and open a notebook of your choice


7. Activate the virtual environment in the notebook:

   `Kernel` -> `Change kernel` -> `venv-nerblackbox`

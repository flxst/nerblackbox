r"""This is the nerblackbox package docstring."""
########################################################################################################################
import os
from os.path import abspath

# create environment variable DATA_DIR with default value if it does not exist
if os.environ.get("DATA_DIR") is None:
    os.environ["DATA_DIR"] = abspath("./store")

########################################################################################################################
from nerblackbox import __about__
from nerblackbox.api.store import Store
from nerblackbox.api.dataset import Dataset
from nerblackbox.api.experiment import Experiment
from nerblackbox.api.model import Model
from nerblackbox.modules.ner_training.data_preprocessing.text_encoder import TextEncoder

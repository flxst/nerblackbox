from configparser import ConfigParser
from os.path import isfile
from nerblackbox.modules.annotation.annotation_tool_doccano import AnnotationToolDoccano
from nerblackbox.modules.annotation.annotation_tool_labelstudio import AnnotationToolLabelStudio
from nerblackbox.modules.annotation.annotation_tool_base import AnnotationToolBase


class AnnotationTool:

    @classmethod
    def from_config(cls,
                    dataset_name: str,
                    config_file: str,
                    verbose: bool = False) -> AnnotationToolBase:
        """
        Args:
            dataset_name: e.g. 'strangnas_test'
            config_file: path to config file
            verbose: output
        """

        assert isfile(config_file), f"config file {config_file} does not exist."
        config = ConfigParser()
        config.read(config_file)
        config_dict = dict(config.items("main"))
        tool = config_dict.pop("tool")
        if verbose:
            print(f"> read config from {config_file}")
            print(f"> tool = {tool}")

        if tool == "doccano":
            return AnnotationToolDoccano(dataset_name, config_dict)
        elif tool == "labelstudio":
            return AnnotationToolLabelStudio(dataset_name, config_dict)
        else:
            raise Exception(f"ERROR! tool = {tool} not implemented.")

from configparser import ConfigParser
from os.path import isfile
from nerblackbox.modules.annotation.annotation_tool_doccano import AnnotationToolDoccano
from nerblackbox.modules.annotation.annotation_tool_labelstudio import AnnotationToolLabelStudio
from nerblackbox.modules.annotation.annotation_tool_base import AnnotationToolBase


class AnnotationTool:

    @classmethod
    def from_config(cls,
                    dataset_name: str,
                    config_file: str) -> AnnotationToolBase:
        """
        Args:
            dataset_name: e.g. 'strangnas_test'
            config_file: path to config file
        """

        assert isfile(config_file), f"config file {config_file} does not exist."
        config = ConfigParser()
        config.read(config_file)
        config_dict = dict(config.items("main"))
        print(f"> read config from {config_file}")
        tool = config_dict.pop("tool")
        print(f"> tool = {tool}")

        if tool == "doccano":
            return AnnotationToolDoccano(dataset_name, config_dict)
        elif tool == "labelstudio":
            return AnnotationToolLabelStudio(dataset_name, config_dict)
        else:
            raise Exception(f"ERROR! tool = {tool} not implemented.")

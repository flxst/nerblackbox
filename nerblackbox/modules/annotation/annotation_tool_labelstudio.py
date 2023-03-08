import json
from typing import Dict, List
from label_studio_sdk import Client as LabelStudioClient
from label_studio_sdk import Project as LabelStudioProject
from nerblackbox.modules.annotation.file_conversion import (
    nerblackbox2labelstudio,
    labelstudio2nerblackbox,
)
from nerblackbox.modules.annotation.utils import extract_labels
from nerblackbox.modules.annotation.annotation_tool_base import AnnotationToolBase
from nerblackbox.modules.annotation.io import read_jsonl, write_jsonl, read_json, write_json


class AnnotationToolLabelStudio(AnnotationToolBase):
    def __init__(self,
                 _dataset_name: str,
                 _config_dict: Dict[str, str]):
        """
        Args:
            _dataset_name: e.g. 'strangnas_test'
            _config_dict: [dict] w/ keys 'url', 'username', 'password'
        """
        url = _config_dict["url"]
        super().__init__(tool="labelstudio",
                         client=LabelStudioClient(url, _config_dict["api_key"]),
                         url=url,
                         dataset_name=_dataset_name)

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def _login(self) -> None:
        try:
            self.client.check_connection()
            self.connected = True
            print(f"> successfully connected to LabelStudio at {self.url}")
        except Exception:
            print(f"ERROR! could not connect to LabelStudio at {self.url}. Is the server running?")
            self.connected = False

    def _get_projects(self, _project_name: str) -> List[LabelStudioProject]:
        """
        Args:
            _project_name: e.g. 'batch_1'

        Returns:
            projects: list of existing projects that match _project_name
        """
        projects = list()
        for project in self.client.list_projects():
            if project.params["title"] == _project_name:
                projects.append(project)
        return projects

    def _tool2nerblackbox(self, _input_file: str, _output_file: str) -> None:
        """
        Args:
            _input_file: e.g. '[..]/batch_1_TOOL.jsonl
            _output_file: e.g. '[..]/batch_1.jsonl
        """
        input_lines = read_json(_input_file)
        output_lines = labelstudio2nerblackbox(input_lines)
        write_jsonl(_output_file, output_lines)

    def _nerblackbox2tool(self, _input_file: str, _output_file: str) -> None:
        """
        Args:
            _input_file: e.g. '[..]/batch_1.jsonl
            _output_file: e.g. '[..]/batch_1_TOOL.jsonl
        """
        input_lines = read_jsonl(_input_file)
        output_lines = nerblackbox2labelstudio(input_lines)
        write_json(_output_file, output_lines)

    def _download(self,
                  _project: LabelStudioProject, _paths: Dict[str, str], _project_name: str, verbose: bool = False
                  ) -> None:
        """
        download data from project to file_path f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl"

        Args:
            _project: Project
            _paths: [dict] w/ keys 'directory', 'file_nerblackbox', 'file_tool'
            _project_name: e.g. 'batch_1'
            verbose: output
        """
        tasks = _project.export_tasks()
        tasks.reverse()
        with open(_paths['file_tool'], "w") as f:
            f.write(json.dumps(tasks, ensure_ascii=False))
        if verbose:
            print(
                f"> download data from project = {_project_name} to {_paths['file_tool']}"
            )

    def _upload(self, _project_name: str, _paths: Dict[str, str], verbose: bool = False) -> None:
        """
        upload data from file_path f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl" to project

        Args:
            _project_name: e.g. 'batch_2'
            _paths: [dict] w/ keys 'directory', 'file_nerblackbox', 'file_tool'
            verbose: output
        """
        # 2. create project
        project = self.client.start_project(
            title=_project_name,
            description="description",
        )
        print(f"> created project {_project_name}")

        # 3. create label
        label_config = '<View>\n    <Labels name="label" toName="text">\n'

        labels = extract_labels(_paths['file_nerblackbox'])
        if len(labels) > 0:
            for (label_name, label_color) in labels:
                label_config += f'        <Label value="{label_name}" background="{label_color}"/>\n'
                if verbose:
                    print(f"> added label '{label_name}' with color {label_color}")
            label_config += '    </Labels>\n    <Text name="text" value="$text"/>\n</View>'
            project.set_params(label_config=label_config)

        # 4. upload data
        project.import_tasks(
            tasks=_paths['file_tool'],
        )
        if verbose:
            print(f"> uploaded file {_paths['file_tool']}")

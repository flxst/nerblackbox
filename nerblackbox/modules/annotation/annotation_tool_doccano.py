import zipfile
import os
from typing import Dict, List
from doccano_client import DoccanoClient
from doccano_client.models.project import Project as DoccanoProject
from doccano_client.models.data_upload import Task as DoccanoTask
from nerblackbox.modules.annotation.file_conversion import (
    nerblackbox2doccano,
    doccano2nerblackbox,
)
from nerblackbox.modules.annotation.utils import extract_labels
from nerblackbox.modules.annotation.annotation_tool_base import AnnotationToolBase
from nerblackbox.modules.annotation.io import read_jsonl, write_jsonl


class AnnotationToolDoccano(AnnotationToolBase):
    def __init__(self,
                 _dataset_name: str,
                 _config_dict: Dict[str, str]):
        """
        Args:
            _dataset_name: e.g. 'strangnas_test'
            _config_dict: [dict] w/ keys 'url', 'username', 'password'
        """
        url = _config_dict["url"]
        super().__init__(tool="doccano",
                         client=DoccanoClient(url),
                         url=url,
                         dataset_name=_dataset_name,
                         username=_config_dict["username"],
                         password=_config_dict["password"])

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    def _login(self) -> None:
        try:
            self.client.login(username=self.username, password=self.password)
            self.connected = True
            print(f"> successfully connected to Doccano at {self.url}")
        except Exception:
            print(f"ERROR! could not connect to Doccano at {self.url}. Is the server running?")
            self.connected = False

    def _get_projects(self, _project_name: str) -> List[DoccanoProject]:
        """
        Args:
            _project_name: e.g. 'batch_1'

        Returns:
            projects: list of existing projects that match _project_name
        """
        projects = list()
        for project in self.client.list_projects():
            if project.name == _project_name:
                projects.append(project)
        return projects

    def _tool2nerblackbox(self, _input_file: str, _output_file: str) -> None:
        """
        Args:
            _input_file: e.g. '[..]/batch_1_TOOL.jsonl
            _output_file: e.g. '[..]/batch_1.jsonl
        """
        input_lines = read_jsonl(_input_file)
        output_lines = doccano2nerblackbox(input_lines)
        write_jsonl(_output_file, output_lines)

    def _nerblackbox2tool(self, _input_file: str, _output_file: str) -> None:
        """
        Args:
            _input_file: e.g. '[..]/batch_1.jsonl
            _output_file: e.g. '[..]/batch_1_TOOL.jsonl
        """
        input_lines = read_jsonl(_input_file)
        output_lines = nerblackbox2doccano(input_lines)
        write_jsonl(_output_file, output_lines)

    def _download(self,
                  _project: DoccanoProject, _paths: Dict[str, str], _project_name: str, verbose: bool = False
                  ) -> None:
        """
        download data from project to file_path f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl"

        Args:
            _project: Project
            _paths: [dict] w/ keys 'directory', 'file_nerblackbox', 'file_tool'
            _project_name: e.g. 'batch_1'
            verbose: output
        """
        downloaded_file_path = self.client.download(
            _project.id,
            "JSONL",
        )
        if verbose:
            print(
                f"> download data from project = {_project.name} to {downloaded_file_path}"
            )

        # 2. unzip
        with zipfile.ZipFile(downloaded_file_path, "r") as zip_ref:
            zip_ref.extractall(_paths['directory'])
        os.remove(downloaded_file_path)
        if verbose:
            print(f"> unzip file to {_paths['file_tool']}")
            print(f"> remove zip file")

    def _upload(self, _project_name: str, _paths: Dict[str, str], verbose: bool = False) -> None:
        """
        upload data from file_path f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl" to project

        Args:
            _project_name: e.g. 'batch_2'
            _paths: [dict] w/ keys 'directory', 'file_nerblackbox', 'file_tool'
            verbose: output
        """
        # 2. create project
        project = self.client.create_project(
            _project_name,
            project_type="SequenceLabeling",
            description="description",
        )
        print(f"> created project {_project_name}")

        # 3. create label
        labels = extract_labels(_paths['file_nerblackbox'])
        for (label_name, label_color) in labels:
            self.client.create_label_type(project.id, "span", label_name, color=label_color)
            if verbose:
                print(f"> added label '{label_name}' with color {label_color}")

        # 4. upload data
        self.client.upload(
            project.id,
            [_paths['file_tool']],
            DoccanoTask.SEQUENCE_LABELING,
            "JSONL",
            "text",
            "label",
        )
        if verbose:
            print(f"> uploaded file {_paths['file_tool']}")

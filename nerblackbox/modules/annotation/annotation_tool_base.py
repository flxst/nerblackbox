from os.path import join
from typing import Dict, Union, Optional, List
from doccano_client import DoccanoClient
from label_studio_sdk import Client as LabelStudioClient
from doccano_client.models.project import Project as DoccanoProject
from label_studio_sdk import Project as LabelStudioProject
from nerblackbox import Store
from abc import ABC, abstractmethod

Client = Union[DoccanoClient, LabelStudioClient]
Project = Union[DoccanoProject, LabelStudioProject]


class AnnotationToolBase(ABC):
    """
    Attributes:
        client: DoccanoClient or LabelStudioClient
        dataset_name: e.g. 'strangnas_test'
    """

    def __init__(
        self,
        tool: str,
        client: Client,
        url: str,
        dataset_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Args:
            tool: 'doccano' or 'labelstudio'
            client: e.g. DoccanoClient or LabelStudioClient
            url: e.g. 'https://localhost:8080'
            dataset_name: e.g. 'strangnas_test'
        """
        self.tool = tool
        self.client = client
        self.url = url
        self.dataset_name = dataset_name
        self.username = username
        self.password = password
        self.connected: bool = False
        self._login()

    ####################################################################################################################
    # ABSTRACT BASE METHODS
    ####################################################################################################################
    @abstractmethod
    def _login(self) -> None:
        pass

    @abstractmethod
    def _get_projects(self, _project_name: str) -> List[Project]:
        """
        Args:
            _project_name: e.g. 'batch_1'

        Returns:
            projects: list of existing projects that match _project_name
        """
        pass

    @abstractmethod
    def _tool2nerblackbox(self, _input_file: str, _output_file: str) -> None:
        """
        Args:
            _input_file: e.g. '[..]/batch_1_TOOL.jsonl
            _output_file: e.g. '[..]/batch_1.jsonl
        """
        pass

    @abstractmethod
    def _nerblackbox2tool(self, _input_file: str, _output_file: str) -> None:
        """
        Args:
            _input_file: e.g. '[..]/batch_1.jsonl
            _output_file: e.g. '[..]/batch_1_TOOL.jsonl
        """
        pass

    @abstractmethod
    def _download(
        self,
        _project: Project,
        _paths: Dict[str, str],
        _project_name: str,
        verbose: bool = False,
    ):
        """
        download data from project to file_path f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl"

        Args:
            _project: Project
            _paths: [dict] w/ keys 'directory', 'file_nerblackbox', 'file_tool'
            _project_name: e.g. 'batch_1'
            verbose: output
        """
        pass

    @abstractmethod
    def _upload(
        self, _project_name: str, _paths: Dict[str, str], verbose: bool = False
    ) -> None:
        """
        upload data from file_path f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl" to project

        Args:
            _project_name: e.g. 'batch_2'
            _paths: [dict] w/ keys 'directory', 'file_nerblackbox', 'file_tool'
        """
        pass

    ####################################################################################################################
    # BASE METHODS
    ####################################################################################################################
    def _get_paths(self, project_name: str) -> Dict[str, str]:
        """
        get paths for dataset_name and project_name

        Args:
            project_name: e.g. 'batch_1'

        Returns:
            paths: [dict] w/ keys 'directory', 'file_nerblackbox', 'file_tool'
        """
        paths = {
            "directory": f"{Store.get_path()}/datasets/{self.dataset_name}",
        }
        paths["file_nerblackbox"] = join(paths["directory"], f"{project_name}.jsonl")

        if self.tool == "doccano":
            paths["file_tool"] = join(
                paths["directory"], f"{project_name}_DOCCANO.jsonl"
            )
        elif self.tool == "labelstudio":
            paths["file_tool"] = join(
                paths["directory"], f"{project_name}_LABELSTUDIO.json"
            )
        else:
            raise Exception(f"ERROR! tool = {self.tool} not implemented.")

        return paths

    def get_file_path(self, project_name: str) -> str:
        """
        get path of local file (nerblackbox format) that corresponds to `dataset_name` and `project_name`

        Args:
            project_name: e.g. 'batch_1'

        Returns:
            file_path: e.g. f"{Store.get_path()}/datasets/<dataset_name>/batch_1.jsonl"
        """
        _paths = self._get_paths(project_name)
        return _paths["file_nerblackbox"]

    def _get_project_by_name(
        self, project_name: str, expected_nr_of_projects: int
    ) -> Optional[Project]:
        """
        - checks that number of existing projects with project_name is equal to expected_nr_of_projects
        - returns project if expected_nr_of_projects == 1

        Args:
            project_name: e.g. 'batch_1'
            expected_nr_of_projects: 0 or 1

        Returns:
            project: Project if expected_nr_of_projects == 1, else None
        """
        assert expected_nr_of_projects in [
            0,
            1,
        ], f"ERROR! expected_nr_of_projects needs to be 0 or 1."

        projects = self._get_projects(project_name)

        assert (
            len(projects) == expected_nr_of_projects
        ), f"ERROR! found #projects = {len(projects)} with name = {project_name}, expected {expected_nr_of_projects}"

        if expected_nr_of_projects == 0:
            return None
        else:  # expected_nr_of_projects == 1
            return projects[0]

    def download(self, project_name: str, verbose: bool = False) -> str:
        """
        download data from project to file_path `f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl"`

        Args:
            project_name: e.g. 'batch_1'
            verbose: output

        Returns:
            file_path: e.g. f"{Store.get_path()}/datasets/<dataset_name>/batch_1.jsonl"
        """
        paths = self._get_paths(project_name)

        # 0. find project
        project = self._get_project_by_name(project_name, 1)

        # 1. download data
        self._download(project, paths, project_name, verbose)

        # 2. translate format from labelstudio to nerblackbox
        self._tool2nerblackbox(paths["file_tool"], paths["file_nerblackbox"])
        if verbose:
            print(f"> translate data to nerblackbox format")
        print(f"> save data at {paths['file_nerblackbox']}")

        return self.get_file_path(project_name)

    def upload(self, project_name: str, verbose: bool = False) -> None:
        """
        upload data from file_path `f"{Store.get_path()}/datasets/<dataset_name>/<project_name>.jsonl"` to project

        Args:
            project_name: e.g. 'batch_2'
            verbose: output
        """
        paths = self._get_paths(project_name)

        # 0. check that project does not exist yet
        _ = self._get_project_by_name(project_name, 0)

        # 1. translate format from nerblackbox to labelstudio
        self._nerblackbox2tool(paths["file_nerblackbox"], paths["file_tool"])
        if verbose:
            print(f"> translated data to annotation tool format")

        # 2. upload
        self._upload(project_name, paths, verbose)

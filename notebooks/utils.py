import shutil
from nerblackbox.modules.annotation.io import read_jsonl, write_jsonl
from nerblackbox.modules.annotation.annotation_tool_base import AnnotationToolBase
from nerblackbox import Dataset


def prepare_raw_data(_dataset_name: str) -> None:
    # download dataset from HuggingFace
    dataset = Dataset(_dataset_name)
    dataset.set_up()

    # train dataset -> annotated data
    input_lines = read_jsonl(f"./store/datasets/{_dataset_name}/train.jsonl")
    for input_line in input_lines:
        input_line["tags"] = []
    write_jsonl(f"./store/datasets/{_dataset_name}/annotated_data.jsonl", input_lines)

    # test dataset -> raw data
    input_lines = read_jsonl(f"./store/datasets/{_dataset_name}/test.jsonl")
    for input_line in input_lines:
        input_line["tags"] = []
    write_jsonl(f"./store/datasets/{_dataset_name}/raw_data.jsonl", input_lines)


def upload_raw_data(_dataset_name: str, _annotation_tool: AnnotationToolBase) -> None:
    # upload annotated and raw data to annotation tool
    _annotation_tool.upload(project_name="annotated_data")
    _annotation_tool.upload(project_name="raw_data")


def simulate_annotation(_dataset_name: str, _annotation_tool: AnnotationToolBase) -> None:
    # train dataset -> annotated data
    _ = shutil.copy2(f"./store/datasets/{_dataset_name}/train.jsonl",
                     f"./store/datasets/{_dataset_name}/annotated_data.jsonl")

    # delete annotation_tool project
    project = _annotation_tool._get_project_by_name("annotated_data", 1)
    _annotation_tool.client.delete_project(project.id)

    # upload annotated to annotation tool
    _annotation_tool.upload(project_name="annotated_data")

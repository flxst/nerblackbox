
from typing import Dict, Tuple, List, Optional
from copy import deepcopy
import pandas as pd
from nerblackbox.modules.ner_training.data_preprocessing.tools.utils import (
    InputExamples,
)


class AnnotationSchemeUtils:

    ####################################################################################################################
    # 1. from data preprocessing
    ####################################################################################################################
    @classmethod
    def convert_annotation_scheme(
            cls,
            input_examples: Dict[str, InputExamples],
            tag_list: List[str],
            annotation_scheme_source: str,
            annotation_scheme_target: str) -> Tuple[Dict[str, InputExamples], List[str]]:
        """
        convert input_examples from annotation_scheme_source to annotation_scheme_target

        Args:
            input_examples: [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                      values = [list] of [InputExample]
            tag_list:       [list] of tags present in the dataset, e.g. ['O', 'PER', ..]
            annotation_scheme_source: [str], e.g. plain / bio
            annotation_scheme_target: [str], e.g. bio / plain

        Returns:
            input_examples_converted: [dict] w/ keys = ['train', 'val', 'test'] or ['predict'] &
                                                values = [list] of [InputExample]
            tag_list_converted:       [list] of tags present in the dataset, e.g. ['O', 'B-PER', ..]
        """
        input_examples_converted = deepcopy(input_examples)
        tag_list_converted = list()
        if annotation_scheme_source == "plain" and annotation_scheme_target == "bio":
            # convert input_examples
            for key in input_examples_converted.keys():
                for input_example_converted in input_examples_converted[key]:
                    input_example_converted.tags = " ".join(cls.convert2bio(input_example_converted.tags.split()))

            # convert tag_list
            for tag in tag_list:
                if tag == "O":
                    tag_list_converted.append(tag)
                else:
                    tag_list_converted.append(f"B-{tag}")
                    tag_list_converted.append(f"I-{tag}")
        elif annotation_scheme_source == "bio" and annotation_scheme_target == "plain":
            # convert input_examples
            for key in input_examples_converted.keys():
                for input_example_converted in input_examples_converted[key]:
                    input_example_converted.tags = " ".join(cls.convert2plain(input_example_converted.tags.split()))

            # convert tag_list
            for tag in tag_list:
                if tag == "O":
                    tag_list_converted.append(tag)
                elif tag.startswith("B-"):
                    tag_list_converted.append(tag.split("-")[-1])
        else:
            raise Exception(f"annotation_scheme_source = {annotation_scheme_source} and "
                            f"annotation_scheme_target = {annotation_scheme_target} not implemented.")

        tag_list_converted = sorted(tag_list_converted)

        return input_examples_converted, tag_list_converted

    @classmethod
    def ensure_completeness_in_case_of_bio_tags(cls, tag_list: List[str]) -> List[str]:
        """
        make sure that there is an "I-*" tag for every "B-*" tag in case of BIO-tags
        ----------------------------------------------------------------------------
        :param tag_list:             e.g. ["B-person", "B-time", "I-person"]
        :return: completed_tag_list: e.g. ["B-person", "B-time", "I-person", "I-time"]
        """
        b_tags = [tag for tag in tag_list if tag.startswith("B")]
        for b_tag in b_tags:
            i_tag = b_tag.replace("B-", "I-")
            if i_tag not in tag_list:
                tag_list.append(i_tag)
        return tag_list

    @classmethod
    def order_tag_list(cls, tag_list: List[str]) -> List[str]:
        return ["O"] + sorted([elem for elem in tag_list if elem != "O"])

    @classmethod
    def convert_tag_list_bio2plain(cls, tag_list_bio: List[str]) -> List[str]:
        tag_list_bio_without_o = [elem for elem in tag_list_bio if elem != "O"]
        return ["O"] + sorted(
            pd.Series(tag_list_bio_without_o)
            .map(lambda x: x.split("-")[-1])
            .drop_duplicates()
            .tolist()
        )

    ####################################################################################################################
    # 2. from ner_metrics
    ####################################################################################################################
    @classmethod
    def _assert_plain_tags(cls, tag_list: List[str]) -> None:
        for tag in tag_list:
            if tag != "O" and (len(tag) > 2 and tag[1] == "-"):
                raise Exception(
                    "ERROR! attempt to convert tags to bio format that already seem to have bio format."
                )

    @classmethod
    def _assert_bio_tags(cls, tag_list: List[str]) -> None:
        for tag in tag_list:
            if tag != "O" and (len(tag) <= 2 or tag[1] != "-"):
                raise Exception(
                    "ERROR! assuming tags to have bio format that seem to have plain format instead."
                )

    @classmethod
    def convert2bio(cls, tag_list: List[str], convert_to_bio=True) -> List[str]:
        """
        - add bio prefixes if tag_list is in plain annotation scheme

        Args:
            tag_list:       e.g. ['O',   'ORG',   'ORG']
            convert_to_bio: whether to cast to bio labels

        Returns:
            bio_tag_list:  e.g. ['O', 'B-ORG', 'I-ORG']
        """
        if convert_to_bio:
            cls._assert_plain_tags(tag_list)
            return cls._convert_tags_plain2bio(tag_list)
        else:
            cls._assert_bio_tags(tag_list)
            return list(tag_list)

    @classmethod
    def _convert_tags_plain2bio(cls, tag_list: List[str]) -> List[str]:
        """
        adds bio prefixes to plain tags

        Args:
            tag_list:     e.g. ['O',   'ORG',   'ORG']

        Returns:
            bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']
        """
        return [
            cls._convert_tag_plain2bio(tag_list[i], previous=tag_list[i - 1] if i > 0 else None)
            for i in range(len(tag_list))
        ]

    @classmethod
    def _convert_tag_plain2bio(cls, tag: str, previous: Optional[str] = None) -> str:
        """
        add bio prefix to plain tag, depending on previous tag

        Args:
            tag:      e.g. 'ORG'
            previous: e.g. 'ORG'

        Returns:
            bio_tag:  e.g. 'I-ORG'
        """
        if tag == "O" or tag.startswith("["):
            return tag
        elif previous is None:
            return f"B-{tag}"
        elif tag != previous:
            return f"B-{tag}"
        else:
            return f"I-{tag}"

    @classmethod
    def convert2plain(cls, tag_list: List[str], convert_to_plain=True) -> List[str]:
        """
        - removes bio prefixes if tag_list is in bio annotation scheme

        Args:
            tag_list:  e.g. ['O', 'B-ORG', 'I-ORG']
            convert_to_plain: whether to cast to plain labels

        Returns:
            tag_list_plain:       e.g. ['O',   'ORG',   'ORG']
        """
        if convert_to_plain:
            cls._assert_bio_tags(tag_list)
            return cls._convert_tags_bio2plain(tag_list)
        else:
            cls._assert_plain_tags(tag_list)
            return list(tag_list)

    @classmethod
    def _convert_tags_bio2plain(cls, bio_tag_list: List[str]) -> List[str]:
        """
        retrieve plain tags by removing bio prefixes

        Args:
            bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']

        Returns:
            tag_list:     e.g. ['O',   'ORG',   'ORG']
        """
        return [elem.split("-")[-1] for elem in bio_tag_list]

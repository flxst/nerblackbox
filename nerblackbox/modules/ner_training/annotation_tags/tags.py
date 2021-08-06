
from typing import List, Optional


class Tags:
    
    def __init__(self, tag_list: List[str]):
        """
        :param tag_list: e.g. ["O", "PER", "O", "ORG"] or ["O", "B-person", "I-person", "O"]
        """
        self.tag_list = tag_list

    ####################################################################################################################
    # 2. from ner_metrics
    ####################################################################################################################
    def _assert_plain_tags(self) -> None:
        for tag in self.tag_list:
            if tag != "O" and (len(tag) > 2 and tag[1] == "-"):
                raise Exception(
                    "ERROR! attempt to convert tags to bio format that already seem to have bio format."
                )

    def _assert_bio_tags(self) -> None:
        for tag in self.tag_list:
            if tag != "O" and (len(tag) <= 2 or tag[1] != "-"):
                raise Exception(
                    "ERROR! assuming tags to have bio format that seem to have plain format instead."
                )

    def convert2bio(self, convert_to_bio=True) -> List[str]:
        """
        - add bio prefixes if tag_list is in plain annotation scheme

        Uses:
            tag_list:       e.g. ['O',   'ORG',   'ORG']

        Args:
            convert_to_bio: whether to cast to bio labels

        Returns:
            bio_tag_list:  e.g. ['O', 'B-ORG', 'I-ORG']
        """
        if convert_to_bio:
            self._assert_plain_tags()
            return self._convert_tags_plain2bio()
        else:
            self._assert_bio_tags()
            return list(self.tag_list)

    def _convert_tags_plain2bio(self) -> List[str]:
        """
        adds bio prefixes to plain tags

        Uses:
            tag_list:     e.g. ['O',   'ORG',   'ORG']

        Returns:
            bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']
        """
        return [
            self._convert_tag_plain2bio(self.tag_list[i], previous=self.tag_list[i - 1] if i > 0 else None)
            for i in range(len(self.tag_list))
        ]

    @staticmethod
    def _convert_tag_plain2bio(tag: str, previous: Optional[str] = None) -> str:
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

    def convert2plain(self, convert_to_plain=True) -> List[str]:
        """
        - removes bio prefixes if tag_list is in bio annotation scheme

        Uses:
            tag_list:  e.g. ['O', 'B-ORG', 'I-ORG']

        Args:
            convert_to_plain: whether to cast to plain labels

        Returns:
            tag_list_plain:       e.g. ['O',   'ORG',   'ORG']
        """
        if convert_to_plain:
            self._assert_bio_tags()
            return self._convert_tags_bio2plain()
        else:
            self._assert_plain_tags()
            return list(self.tag_list)

    def _convert_tags_bio2plain(self) -> List[str]:
        """
        retrieve plain tags by removing bio prefixes

        Uses:
            bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']

        Returns:
            tag_list:     e.g. ['O',   'ORG',   'ORG']
        """
        return [elem.split("-")[-1] for elem in self.tag_list]


from typing import List, Optional


class Tags:
    
    def __init__(self, tag_list: List[str]):
        """
        :param tag_list: e.g. ["O", "PER", "O", "ORG"] or ["O", "B-person", "I-person", "O"]
        """
        self.tag_list = tag_list

    ####################################################################################################################
    # 1. MAIN FUNCTIONS
    ####################################################################################################################
    def convert_scheme(self, source_scheme: str, target_scheme: str) -> List[str]:
        """
        Args:
            source_scheme: e.g. 'plain', 'bio'
            target_scheme: e.g. 'bio', 'plain'

        Uses:
            tag_list:       e.g. ['O',   'ORG',   'ORG'] or ['O', 'B-ORG', 'I-ORG']

        Returns:
            tag_list_plain: e.g. ['O', 'B-ORG', 'I-ORG'] or ['O',   'ORG',   'ORG']
        """
        if source_scheme == "plain":
            self._assert_plain_tags()
        elif source_scheme == "bio":
            self._assert_bio_tags()
        else:
            raise Exception(f"ERROR! source scheme = {source_scheme} not implemented.")

        if source_scheme == "plain" and target_scheme == "plain":
            return list(self.tag_list)
        elif source_scheme == "bio" and target_scheme == "bio":
            return self._restore_annotation_scheme_consistency(scheme="bio")
        elif source_scheme == "plain" and target_scheme == "bio":
            return self._convert_tags_plain2bio()
        elif source_scheme == "bio" and target_scheme == "plain":
            return self._convert_tags_bio2plain()
        else:
            raise Exception(f"ERROR! source & target scheme = {source_scheme} & {target_scheme} not implemented.")

    ####################################################################################################################
    # 2. HELPER FUNCTIONS
    ####################################################################################################################
    def _assert_plain_tags(self) -> None:
        for tag in self.tag_list:
            if tag != "O" and (len(tag) > 2 and tag[1] == "-"):
                raise Exception(
                    "ERROR! attempt to convert tags to bio format that already seem to have bio format."
                )

    def _assert_bio_tags(self) -> None:
        for tag in self.tag_list:
            if tag != "O" and (len(tag) <= 2 or tag[0] not in ["B", "I"] or tag[1] != "-"):
                raise Exception(
                    "ERROR! assuming tags to have bio format that seem to have plain format instead."
                )

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
        if tag == "O":
            return tag
        elif previous is None:
            return f"B-{tag}"
        elif tag != previous:
            return f"B-{tag}"
        else:
            return f"I-{tag}"

    def _convert_tags_bio2plain(self) -> List[str]:
        """
        retrieve plain tags by removing bio prefixes

        Uses:
            bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']

        Returns:
            tag_list:     e.g. ['O',   'ORG',   'ORG']
        """
        return [elem.split("-")[-1] for elem in self.tag_list]

    def _restore_annotation_scheme_consistency(self, scheme: str):
        if scheme == "bio":
            return [
                self._convert_tag_bio2bio(self.tag_list[i], previous=self.tag_list[i - 1] if i > 0 else None)
                for i in range(len(self.tag_list))
            ]
        else:
            raise Exception(f"ERROR! restore annotation scheme consistency not implemented for scheme = {scheme}.")

    @staticmethod
    def _convert_tag_bio2bio(current: str, previous: Optional[str] = None) -> str:
        """
        correct bio prefix, depending on previous tag

        Args:
            current:  e.g. 'I-ORG'
            previous: e.g. 'O'

        Returns:
            current_corrected:  e.g. 'B-ORG'
        """
        if current == "O" or current.startswith("B-"):
            return current
        elif current.startswith("I-"):
            if previous is None or previous.split("-")[-1] != current.split("-")[-1]:
                return current.replace("I-", "B-")
            else:
                return current
        else:
            raise Exception(f"ERROR! bio tag {current} should be of the format O, B-*, I-*")


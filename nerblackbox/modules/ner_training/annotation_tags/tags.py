import numpy as np
from typing import List, Optional, Tuple


class Tags:
    def __init__(self, tag_list: List[str]):
        """
        :param tag_list: e.g. ["O", "PER", "O", "ORG"] or ["O", "B-person", "I-person", "O"]
        """
        self.tag_list = tag_list

    ####################################################################################################################
    # 1. MAIN METHODS
    ####################################################################################################################
    def convert_scheme(self, source_scheme: str, target_scheme: str) -> List[str]:
        """
        Args:
            source_scheme: e.g. 'plain', 'bio'
            target_scheme: e.g. 'bio', 'plain'

        Uses:
            tag_list:           e.g. ['O',   'ORG',   'ORG'] or ['O', 'B-ORG', 'I-ORG']

        Returns:
            tag_list_converted: e.g. ['O', 'B-ORG', 'I-ORG'] or ['O',   'ORG',   'ORG']
        """
        if source_scheme == "plain":
            self._assert_plain_tags()
        elif source_scheme == "bio":
            self._assert_bio_tags()
        elif source_scheme == "bilou":
            self._assert_bilou_tags()
        else:
            raise Exception(f"ERROR! source scheme = {source_scheme} not implemented.")

        if source_scheme == target_scheme:  # "plain" and target_scheme == "plain":
            return list(self.tag_list)
        elif source_scheme == "plain" and target_scheme == "bio":
            return self._convert_tags_plain2bio()
        elif source_scheme == "plain" and target_scheme == "bilou":
            return self._convert_tags_plain2bilou()
        elif source_scheme in ["bio", "bilou"] and target_scheme == "plain":
            return self._convert_tags_bio2plain()
        elif source_scheme == "bio" and target_scheme == "bilou":
            return self.restore_annotation_scheme_consistency(scheme=target_scheme)[
                0
            ]  # conversion
        elif source_scheme == "bilou" and target_scheme == "bio":
            return self._convert_tags_bilou2bio()
        else:
            raise Exception(
                f"ERROR! source & target scheme = {source_scheme} & {target_scheme} not implemented."
            )

    def get_annotation_scheme_abidance(
        self, scheme: str
    ) -> float:  # Tuple[int, int, float]:
        """
        compute how many tokens in self.tag_list abide to the annotation scheme

        Returns:
            annotation_scheme_abidance:
                - absolute number of correct tokens
                - total number of tokens
                - relative number of correct tokens
        """
        # return 49.0, 50.0, 0.98
        return 0.98

    ####################################################################################################################
    # 2. HELPER METHODS
    ####################################################################################################################
    def _assert_plain_tags(self) -> None:
        for tag in self.tag_list:
            if tag != "O" and (len(tag) > 2 and tag[1] == "-"):
                raise Exception("ERROR! tags to not have expected plain format.")

    def _assert_bio_tags(self) -> None:
        for tag in self.tag_list:
            if tag != "O" and (
                len(tag) <= 2 or tag[0] not in ["B", "I"] or tag[1] != "-"
            ):
                raise Exception("ERROR! tags to not have expected bio format.")

    def _assert_bilou_tags(self) -> None:
        for tag in self.tag_list:
            if tag != "O" and (
                len(tag) <= 2 or tag[0] not in ["B", "I", "L", "U"] or tag[1] != "-"
            ):
                raise Exception("ERROR! tags to not have expected bilou format.")

    def _convert_tags_plain2bio(self) -> List[str]:
        """
        adds bio prefixes to plain tags

        Uses:
            tag_list:     e.g. ['O',   'ORG',   'ORG']

        Returns:
            bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']
        """
        return [
            self._convert_tag_plain2bio(
                self.tag_list[i], previous=self.tag_list[i - 1] if i > 0 else None
            )
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
        elif previous is None or tag != previous:
            return f"B-{tag}"
        else:
            return f"I-{tag}"

    def _convert_tags_plain2bilou(self) -> List[str]:
        """
        adds bilou prefixes to plain tags

        Uses:
            tag_list:     e.g. ['O',   'ORG',   'ORG']

        Returns:
            bio_tag_list: e.g. ['O', 'B-ORG', 'I-ORG']
        """
        return [
            self._convert_tag_plain2bilou(
                self.tag_list[i],
                previous=self.tag_list[i - 1] if i > 0 else None,
                subsequent=self.tag_list[i + 1] if i < len(self.tag_list) - 1 else None,
            )
            for i in range(len(self.tag_list))
        ]

    @staticmethod
    def _convert_tag_plain2bilou(
        tag: str, previous: Optional[str] = None, subsequent: Optional[str] = None
    ) -> str:
        """
        add bilou prefix to plain tag, depending on previous and subsequent tag

        Args:
            tag:        e.g. 'ORG'
            previous:   e.g. 'ORG'
            subsequent: e.g. 'ORG'

        Returns:
            bilou_tag:  e.g. 'I-ORG'
        """
        if tag == "O":
            return tag
        else:
            previous_condition = previous is None or tag != previous
            subsequent_condition = subsequent is None or tag != subsequent
            if previous_condition and subsequent_condition:
                return f"U-{tag}"
            elif previous_condition and not subsequent_condition:
                return f"B-{tag}"
            elif not previous_condition and subsequent_condition:
                return f"L-{tag}"
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

    def _convert_tags_bilou2bio(self) -> List[str]:
        """
        convert bilou to bio tags

        Uses:
            bilou_tag_list: e.g. ['O', 'B-ORG', 'L-ORG']

        Returns:
            bio_tag_list:   e.g. ['O', 'B-ORG', 'I-ORG']
        """
        return [self._convert_tag_bilou2bio(elem) for elem in self.tag_list]

    @staticmethod
    def _convert_tag_bilou2bio(tag: str):
        """
        Args:
            tag:      e.g. 'L-ORG'

        Returns:
            bio_tag:  e.g. 'I-ORG'
        """
        if tag == "O" or tag.startswith("B-") or tag.startswith("I-"):
            return tag
        else:
            plain = tag.split("-")[-1]
            if tag.startswith("U-"):
                return f"B-{plain}"
            elif tag.startswith("L-"):
                return f"I-{plain}"

    def restore_annotation_scheme_consistency(
        self, scheme: str
    ) -> Tuple[List[str], float]:
        """

        Args:
            scheme:

        Uses:
            tag_list:           e.g. ['O',   'ORG',   'ORG'] or ['O', 'B-ORG', 'I-ORG']

        Returns:
            tag_list_converted: e.g. ['O', 'B-ORG', 'I-ORG'] or ['O',   'ORG',   'ORG']
            asr_abidance:       e.g. 0.98
                = how many tokens in self.tag_list abide to the annotation scheme
                = absolute number of correct tokens / total number of tokens
        """
        if scheme == "bio":
            converted_tuples = [
                self.convert_tag_bio2bio(
                    self.tag_list[i], previous=self.tag_list[i - 1] if i > 0 else None
                )
                for i in range(len(self.tag_list))
            ]
        elif scheme == "bilou":
            converted_tuples = [
                self.convert_tag_bilou2bilou(
                    self.tag_list[i],
                    previous=self.tag_list[i - 1] if i > 0 else None,
                    subsequent=self.tag_list[i + 1]
                    if i < len(self.tag_list) - 1
                    else None,
                )
                for i in range(len(self.tag_list))
            ]
        else:
            raise Exception(
                f"ERROR! restore annotation scheme consistency not implemented for scheme = {scheme}."
            )

        tag_list_converted = [tup[0] for tup in converted_tuples]
        asr_abidance = 1 - np.average([tup[1] for tup in converted_tuples])
        return tag_list_converted, asr_abidance

    ####################################################################################################################
    # 3. CLASS METHODS
    ####################################################################################################################
    @classmethod
    def convert_tag_bio2bio(
        cls, current: str, previous: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        correct bio prefix, depending on previous tag

        Args:
            current:  e.g. 'I-ORG'
            previous: e.g. 'O'

        Returns:
            current_corrected:  e.g. 'B-ORG'
            bool_corrected:     e.g. True
        """
        if current == "O" or current.startswith("B-"):
            return current, False
        else:
            assert (
                current[0] in ["I"] and current[1] == "-"
            ), f"ERROR! bio tag {current} should be of the format O, B-*, I-*"

            plain = current.split("-")[-1]
            condition_previous = previous is None or previous not in [
                f"B-{plain}",
                f"I-{plain}",
            ]

            if condition_previous:
                # rule 1: !I/!B + I -> B
                return f"B-{plain}", True
            else:
                return f"I-{plain}", False

    @classmethod
    def convert_tag_bilou2bilou(
        cls,
        current: str,
        previous: Optional[str] = None,
        subsequent: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        correct bilou prefix, depending on previous and next tag

        Args:
            current:    e.g. 'I-ORG'
            previous:   e.g. 'O'
            subsequent: e.g. 'O'

        Returns:
            current_corrected:  e.g. 'U-ORG'
            bool_corrected:     e.g. True
        """
        if current == "O" or current.startswith("U-"):
            return current, False
        else:
            assert (
                current[0] in ["B", "I", "L"] and current[1] == "-"
            ), f"ERROR! bilou tag {current} should be of the format O, B-*, I-*, L-*, U-*"

            plain = current.split("-")[-1]
            condition_previous = previous is None or previous not in [
                f"B-{plain}",
                f"I-{plain}",
            ]
            condition_subsequent = subsequent is None or subsequent not in [
                f"I-{plain}",
                f"L-{plain}",
            ]
            bool_corrected = False
            while 1:
                if current.startswith("I-"):
                    if condition_subsequent:
                        # rule 1: I + !I/!L -> L
                        current = f"L-{plain}"
                        bool_corrected = True
                    elif condition_previous:
                        # rule 2: !B/!I + I -> B
                        current = f"B-{plain}"
                        bool_corrected = True
                    else:
                        break
                elif current.startswith("L-"):
                    if condition_previous:
                        # rule 3: !B/!I + L -> B
                        current = f"B-{plain}"
                        bool_corrected = True
                    else:
                        break
                elif current.startswith("B-"):
                    if condition_subsequent:
                        # rule 4: B + !I/!L -> U
                        current = f"U-{plain}"
                        bool_corrected = True
                    else:
                        break
                elif current.startswith("U-"):
                    break
                else:
                    raise Exception(
                        f"ERROR! this should not happen. current = {current}"
                    )

            return current, bool_corrected

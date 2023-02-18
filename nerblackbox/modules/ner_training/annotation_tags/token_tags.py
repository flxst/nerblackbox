from typing import List, Dict, Union, Any, Tuple
from nerblackbox.modules.ner_training.annotation_tags.tags import Tags
from copy import deepcopy


class TokenTags:
    def __init__(
        self, token_tag_list: List[Dict[str, str]], scheme: str, level: str = "token"
    ):
        """
        Args:
            token_tag_list: e.g. [
                {"char_start": "0", "char_end": "7", "token": "example", "tag": "I-TAG"},
                ..
            ]
            scheme: e.g. "plain" or "bio"
        """
        self.token_tag_list: List[Dict[str, str]] = token_tag_list
        self.scheme = scheme
        self.level = level

        if self.level in ["token", "word"]:
            self._assert_scheme_consistency()

    def _assert_scheme_consistency(self):
        """
        assert that tags in self.token_tag_list are in accordance with self.scheme
        """
        tags = [elem["tag"] for elem in self.token_tag_list if elem["tag"] != "O"]
        if len(tags) == 0:
            possible_schemes = ["plain", "bio", "bilou"]
        elif all(["-" not in elem for elem in tags]):
            possible_schemes = ["plain"]
        elif all(["-" in elem for elem in tags]):
            possible_schemes = ["bio", "bilou"]
        else:
            raise Exception(
                "ERROR! inconsistent tags found. they do not seem to belong to a well-defined scheme."
            )

        assert (
            self.scheme in possible_schemes
        ), f"ERROR! scheme = {self.scheme} is inconsistent with possible_schemes = {possible_schemes}!"

    def as_list(self):
        return self.token_tag_list

    ####################################################################################################################
    # MAIN METHODS
    ####################################################################################################################
    def restore_annotation_scheme_consistency(self) -> None:
        """
        plain tags: not modified
        bio   tags: restore annotation scheme consistency
        bilou tags: restore annotation scheme consistency

        Changed Attr:
            token_tag_list: List[Dict[str, str]], e.g.
            [
                {"char_start": "0", "char_end": "7", "token": "example", "tag": "I-TAG"},
                ..
            ]
            --->
            [
                {"char_start": "0", "char_end": "7", "token": "example", "tag": "B-TAG"},
                ..
            ]
        """
        if self.scheme == "plain":
            pass
        elif self.scheme in ["bio", "bilou"]:
            token_tag_list_restored: List[Dict[str, str]] = list()
            for i in range(len(self.token_tag_list)):
                example_word_prediction_restored = self.token_tag_list[i]
                current_tag = self.token_tag_list[i]["tag"]
                current_tag = self._assert_str(current_tag, "current_tag")
                previous_tag = self.token_tag_list[i - 1]["tag"] if i > 0 else None

                if self.scheme == "bio":
                    example_word_prediction_restored["tag"] = Tags.convert_tag_bio2bio(
                        current_tag, previous=previous_tag
                    )[0]
                else:
                    subsequent_tag = (
                        self.token_tag_list[i + 1]["tag"]
                        if i < len(self.token_tag_list) - 1
                        else None
                    )
                    example_word_prediction_restored[
                        "tag"
                    ] = Tags.convert_tag_bilou2bilou(
                        current_tag, previous=previous_tag, subsequent=subsequent_tag
                    )[
                        0
                    ]

                token_tag_list_restored.append(example_word_prediction_restored)

            assert len(token_tag_list_restored) == len(self.token_tag_list), f"ERROR!"

            self.token_tag_list = token_tag_list_restored
        else:
            raise Exception(
                f"ERROR! restore annotation scheme consistency "
                f"not implemented for scheme = {self.scheme}."
            )

    def merge_tokens_to_words(self) -> None:
        """
        discard tokens that are not first token of a word

        Changed Attr:
            token_tag_list: List[Dict[str, str]], e.g.
            [
                {"char_start": "0", "char_end": "4", "token": "2020", "tag": "B-TAG"},
                {"char_start": "4", "char_end": "5", "token": "-, "tag": "I-TAG"},
                {"char_start": "5", "char_end": "7", "token": "04, "tag": "I-TAG"},
                ..
            ]
            --->
            [
                {"char_start": "0", "char_end": "7", "token": "2020-04", "tag": "B-TAG"},
                ..
            ]
            level: 'word'

        """
        for i in range(len(self.token_tag_list) - 1, 0, -1):
            if (
                i > 0
                and self.token_tag_list[i]["char_start"]
                == self.token_tag_list[i - 1]["char_end"]
            ):
                self.token_tag_list[i]["tag"] = "DELETE"
                self.token_tag_list[i - 1]["char_end"] = self.token_tag_list[i][
                    "char_end"
                ]
                self.token_tag_list[i - 1]["token"] += self.token_tag_list[i]["token"]

        self.token_tag_list = [
            elem for elem in self.token_tag_list if elem["tag"] != "DELETE"
        ]
        self.level = "word"

    def unpretokenize(self, _pretokenization_offsets: List[Tuple[int, int]]):
        """
        revert pretokenization using pretokenization offsets.

        Args:
            _pretokenization_offsets: e.g. [(0,4), (4,5), (5,7), (7,8), (8,10), (11,15)]

        Changed Attr:
            token_tag_list: List[Dict[str, str]], e.g.
                [
                    {'char_start': '0', 'char_end': '4', 'token': '2021', 'tag': 'B-PI'},
                    {'char_start': '5', 'char_end': '6', 'token': '-', 'tag': 'I-PI'},
                    {'char_start': '7', 'char_end': '9', 'token': '10', 'tag': 'I-PI'},
                    {'char_start': '10', 'char_end': '11', 'token': '-', 'tag': 'I-PI'},
                    {'char_start': '12', 'char_end': '14', 'token': '14', 'tag': 'I-PI'},
                    {'char_start': '15', 'char_end': '20', 'token': 'Mamma', 'tag': 'O'},
                ]
            --->
                [
                    {'char_start': '0', 'char_end': '4', 'token': '2021', 'tag': 'B-PI'},
                    {'char_start': '4', 'char_end': '5', 'token': '-', 'tag': 'I-PI'},
                    {'char_start': '5', 'char_end': '7', 'token': '10', 'tag': 'I-PI'},
                    {'char_start': '7', 'char_end': '8', 'token': '-', 'tag': 'I-PI'},
                    {'char_start': '8', 'char_end': '10', 'token': '14', 'tag': 'I-PI'},
                    {'char_start': '11', 'char_end': '15', 'token': 'Mamma', 'tag': 'O'},
                ]
        """
        assert len(self.token_tag_list) == len(_pretokenization_offsets), (
            f"ERROR! #token_tag_list = {len(self.token_tag_list)} != "
            f"#pretokenization_offsets = {len(_pretokenization_offsets)}"
        )

        nr_tokens = len(self.token_tag_list)
        for j in range(nr_tokens):
            self.token_tag_list[j]["char_start"] = str(_pretokenization_offsets[j][0])
            self.token_tag_list[j]["char_end"] = str(_pretokenization_offsets[j][1])

    def merge_tokens_to_entities(self, original_text: str, verbose: bool) -> None:
        """
        plain tags:
            - discard tokens with tag 'O'
        bio   tags:
            - merge token predictions that belong together (B-* & I-*)
            - discard tokens with tag 'O'
        bilou tags:
            - merge token predictions that belong together (B-* & I-* & L-*; U-*)
            - discard tokens with tag 'O'

        Args:
            original_text: e.g. 'example sentence.'
            verbose: e.g. False

        Changed Attr:
            token_tag_list: List[Dict[str, str]], e.g.
            [
                {"char_start": "0", "char_end": "7", "token": "example", "tag": "B-TAG"},
                {"char_start": "8", "char_end": "16", "token": "sentence", "tag": "I-TAG"},
                {"char_start": "17", "char_end": "18", "token": ".", "tag": "O"},
                ..
            ]
            --->
            [
                {"char_start": "0", "char_end": "16", "token": "example sentence", "tag": "TAG"},
                ..
            ]
            level: 'entity'
        """
        count = {
            "o_tags": 0,
            "replace": 0,
            "drop": 0,
            "merge": 0,
            "unmodified": 0,
        }
        entity_prefixes_without_b = {
            "bio": ["I-"],
            "bilou": ["I-", "L-"],
        }

        threshold = 0
        merged_ner_tags = list()
        for i in range(len(self.token_tag_list)):
            current_tag = self.token_tag_list[i]["tag"]
            current_tag = self._assert_str(current_tag, "current_tag")
            n_tags = 0
            if current_tag == "O":
                count["o_tags"] += 1
            else:
                merged_ner_tag = None
                if i >= threshold:
                    if self.scheme == "plain":
                        for n in range(i + 1, len(self.token_tag_list)):
                            subsequent_tag = self.token_tag_list[n]["tag"]
                            subsequent_tag = self._assert_str(
                                subsequent_tag, "subsequent_tag"
                            )
                            if subsequent_tag == current_tag:
                                n_tags += 1
                            else:
                                threshold = n
                                break
                            if n == len(self.token_tag_list) - 1:
                                threshold = n + 1  # such that last element is dropped
                        merged_ner_tag = self._merge_tokens(i, original_text, n_tags)
                    elif self.scheme in ["bio", "bilou"]:
                        if current_tag.startswith("B-"):  # BIO scheme
                            plain = current_tag.split("-")[-1]
                            for n in range(i + 1, len(self.token_tag_list)):
                                subsequent_tag = self.token_tag_list[n]["tag"]
                                subsequent_tag = self._assert_str(
                                    subsequent_tag, "subsequent_tag"
                                )
                                if (
                                    len(subsequent_tag) > 2
                                    and subsequent_tag[:2]
                                    in entity_prefixes_without_b[self.scheme]
                                    and subsequent_tag[2:] == plain
                                ):
                                    n_tags += 1
                                    if subsequent_tag[:2] in [
                                        "L-"
                                    ]:  # only applies to "bilou"
                                        # end of entity
                                        threshold = n + 1
                                        break
                                else:
                                    # end of entity
                                    threshold = n
                                    break
                                if n == len(self.token_tag_list) - 1:
                                    threshold = (
                                        n + 1
                                    )  # such that last element is dropped
                            merged_ner_tag = self._merge_tokens(
                                i, original_text, n_tags
                            )
                        elif current_tag.startswith("I-"):
                            count["drop"] += 1
                        elif current_tag.startswith("L-"):  # only applies to "bilou"
                            count["drop"] += 1
                        elif current_tag.startswith("U-"):  # only applies to "bilou"
                            merged_ner_tag = self._merge_tokens(
                                i, original_text, n_tags
                            )

                if merged_ner_tag is not None:
                    merged_ner_tags.append(merged_ner_tag)
                    if n_tags == 0:
                        if self.scheme == "plain":
                            count["unmodified"] += 1
                        else:
                            count["replace"] += 1
                    else:
                        count["merge"] += 1 + n_tags

        assert count["o_tags"] + count["replace"] + count["drop"] + count[
            "merge"
        ] + count["unmodified"] == len(
            self.token_tag_list
        ), f"{count} -> {sum(count.values())} != {len(self.token_tag_list)} | {self.token_tag_list}"

        """
        if count["merge"] > 0 or count["replace"] > 0:
            assert (
                    count["unmodified"] == 0
            ), f"{count} -> if merge or replaced are > 0, unmodified should be == 0."

        if count["unmodified"] > 0:
            assert (
                    count["merge"] == 0
            ), f"{count} -> if unmodified is > 0, merge should be == 0."
            assert (
                    count["replace"] == 0
            ), f"{count} -> if unmodified is > 0, replace should be == 0."
        """

        self.token_tag_list = merged_ner_tags
        if verbose:
            print(
                f"> merged {len(self.token_tag_list)} BIO-tags "
                f"(simple replace: {count['replace']}, "
                f"merge: {count['merge']}, "
                f"O-tags: {count['o_tags']}, "
                f"drop: {count['drop']}, "
                f"unmodified: {count['unmodified']}).\n"
            )

        self.level = "entity"

    ####################################################################################################################
    # HELPER
    ####################################################################################################################
    @staticmethod
    def _assert_str(_object: Union[str, Dict[Any, Any]], _object_name: str) -> str:
        assert isinstance(
            _object, str
        ), f"ERROR! {_object_name} = {_object} is {type(_object)} but should be string"
        return _object

    def _merge_tokens(
        self, _index: int, _original_text: str, _n_tags: int
    ) -> Dict[str, str]:
        """
        merge tokens [_index: _index+_ntags] using self.token_tag_list

        Args:
            _index:         e.g. 1
            _original_text: e.g. 'annotera den här texten'
            _n_tags:        e.g. 1

        Return:
            merged_token: e.g. {'char_start': '9', 'char_end': '16', 'token': 'den här', 'tag': 'ORG'}
        """
        merged_ner_tag = deepcopy(self.token_tag_list[_index])
        assert isinstance(
            merged_ner_tag["tag"], str
        ), "ERROR! merged_ner_tag.tag should be a string"

        # 1. replace tag
        merged_ner_tag["tag"] = merged_ner_tag["tag"].split("-")[-1]

        if _n_tags > 0:
            # 2. replace char_end
            merged_ner_tag["char_end"] = self.token_tag_list[_index + _n_tags][
                "char_end"
            ]

            assert isinstance(
                merged_ner_tag["char_start"], str
            ), "ERROR! merged_ner_tag.char_start should be a string"
            assert isinstance(
                merged_ner_tag["char_end"], str
            ), "ERROR! merged_ner_tag.char_end should be a string"

            # 3. replace token
            merged_ner_tag["token"] = _original_text[
                int(merged_ner_tag["char_start"]) : int(merged_ner_tag["char_end"])
            ]
        return merged_ner_tag

import json

import numpy as np
from transformers import AutoModelForTokenClassification
from torch.nn.functional import softmax
from typing import List, Union, Tuple, Any, Dict, IO
from omegaconf import OmegaConf

from nerblackbox.modules.ner_training.ner_model import NerModel

from transformers import logging

# TODO: this suppresses warnings (https://github.com/huggingface/transformers/issues/5421), root problem should be fixed
logging.set_verbosity_error()

VERBOSE = False


class NerModelPredict(NerModel):
    """
    class that predicts tags for given text
    """

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Union[str, IO]) -> "NerModelPredict":
        """load model in inference mode from checkpoint_path

        Args:
            checkpoint_path: path to checkpoint

        Returns:
            model loaded from checkpoint
        """
        model = super().load_from_checkpoint(checkpoint_path)
        model.freeze()  # for inference mode
        return model

    def __init__(self, hparams: OmegaConf):
        """
        Args:
            hparams: attr experiment_name, run_name, pretrained_model_name, dataset_name, ..
        """
        super().__init__(hparams)

    ####################################################################################################################
    # Abstract Base Methods ############################################################################################
    ####################################################################################################################
    def _preparations(self):
        """
        :created attr: default_logger    [DefaultLogger]
        :created attr: logged_metrics    [LoggedMetrics]
        :created attr: tokenizer         [transformers AutoTokenizer]
        :created attr: data_preprocessor [DataPreprocessor]
        :created attr: tag_list          [list] of tags in dataset, e.g. ['O', 'PER', 'LOC', ..]
        :created attr: model             [transformers AutoModelForTokenClassification]
        :return: -
        """
        # predict
        self._preparations_predict()  # attr: default_logger
        self._preparations_data_general()  # attr: tokenizer, data_preprocessor
        self._preparations_data_predict()  # attr: tag_list, model

    def _preparations_predict(self):
        """
        :created attr: default_logger         [None]
        :return: -
        """
        self.default_logger = None

    def _preparations_data_predict(self):
        """
        :created attr: tag_list          [list] of tags in dataset, e.g. ['O', 'PER', 'LOC', ..]
        :created attr: model             [transformers AutoModelForTokenClassification]
        :return: -
        """
        # tag_list
        self.tag_list = json.loads(self.hparams.tag_list)

        # model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=len(self.tag_list),
            return_dict=False,
        )
        self.model.resize_token_embeddings(
            len(self.tokenizer)
        )  # due to addtional_special_tokens = NEWLINE_TOKENS

    ####################################################################################################################
    # PREDICT
    ####################################################################################################################
    def predict(
        self,
        input_texts: Union[str, List[str]],
        level: str = "entity",
        autocorrect: bool = True,
    ) -> List[List[Dict[str, str]]]:
        """predict tags for input texts. output on entity or word level.

        Examples:
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="entity")
            # [
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "LOC"},
            # ]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="word", autocorrect=False)
            # [
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "I-ORG"},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
            #     {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-LOC"},
            # ]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="word", autocorrect=True)
            # [
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "B-ORG"},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
            #     {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-LOC"},
            # ]
            ```

        Args:
            input_texts:   e.g. ["example 1", "example 2"]
            level:         "entity" or "word"
            autocorrect:   if True, autocorrect annotation scheme (e.g. B- and I- tags). needs to be True if level == "entity".

        Returns:
            predictions: [list] of predictions for the different examples.
                         each list contains a [list] of [dict] w/ keys = char_start, char_end, word, tag
        """
        return self._predict(input_texts, level, autocorrect, proba=False)

    def predict_proba(
        self, input_texts: Union[str, List[str]]
    ) -> List[List[Dict[str, Union[str, Dict]]]]:
        """predict probability distributions for input texts. output on word level.

        Examples:
            ```
            predict_proba(["arbetsförmedlingen finns i stockholm"])
            # [
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "proba_dist: {"O": 0.21, "B-ORG": 0.56, ..}},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "proba_dist: {"O": 0.87, "B-ORG": 0.02, ..}},
            #     {"char_start": "25", "char_end": "26", "token": "i", "proba_dist: {"O": 0.95, "B-ORG": 0.01, ..}},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "proba_dist: {"O": 0.14, "B-ORG": 0.22, ..}},
            # ]
            ```

        Args:
            input_texts:   e.g. ["example 1", "example 2"]

        Returns:
            predictions: [list] of probability predictions for different examples.
                         each list contains a [list] of [dict] w/ keys = char_start, char_end, word, proba_dist
                         where proba_dist = [dict] that maps self.tag_list to probabilities
        """
        return self._predict(input_texts, level="word", autocorrect=False, proba=True)

    def _predict(
        self,
        input_texts: Union[str, List[str]],
        level: str = "entity",
        autocorrect: bool = True,
        proba: bool = False,
    ) -> List[List[Dict[str, Union[str, Dict]]]]:
        """predict tags or probabilities for tags

        Args:
            input_texts:  e.g. ["example 1", "example 2"]
            level:        "entity" or "word"
            autocorrect:  if True, autocorrect annotation scheme (e.g. B- and I- tags). needs to be True if level == "entity".
            proba:        if True, predict probabilities instead of labels (on word level)

        Returns:
            predictions: [list] of [list] of [dict] w/ keys = char_start, char_end, word, tag/proba_dist
                         where proba_dist = [dict] that maps self.tag_list to probabilities
        """
        # --- check input arguments ---
        assert level in [
            "entity",
            "word",
        ], f"ERROR! model prediction level = {level} unknown, needs to be entity or word."
        if level == "entity":
            assert (
                autocorrect is True
            ), f"ERROR! level = entity requires autocorrect = True."
        if proba:
            assert level == "word" and autocorrect is False, (
                f"ERROR! probability predictions require level = word and autocorrect = False. "
                f"level = {level} and autocorrect = {autocorrect} not possible."
            )
        # ------------------------------

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        predictions = list()
        for input_text in input_texts:
            predict_dataloader = self._get_predict_dataloader([input_text])

            if self.data_preprocessor.do_lower_case:
                input_text = input_text.lower()

            input_text_tokens = list()
            input_text_token_predictions = list()
            ######################################
            # 1 input_text, individual chunks, tokens
            ######################################
            for sample in predict_dataloader:
                token_tensors, tokens = self._predict_on_tokens(sample)
                if proba is False:
                    token_predictions = self._turn_tensors_into_tags(token_tensors)
                else:
                    token_predictions = (
                        self._turn_tensors_into_tag_probability_distributions(
                            token_tensors
                        )
                    )
                input_text_tokens.extend(tokens)
                input_text_token_predictions.extend(token_predictions)

            ######################################
            # 1 input_text, merged chunks, tokens -> words
            ######################################
            _input_text_word_predictions: List[
                Tuple[Union[str, Any], ...]
            ] = get_tags_on_words_between_special_tokens(
                input_text_tokens, input_text_token_predictions
            )

            input_text_word_predictions: List[
                Dict[str, Union[str, Dict]]
            ] = restore_unknown_tokens(_input_text_word_predictions, input_text)

            if autocorrect:
                input_text_word_predictions = restore_annotation_scheme_consistency(
                    input_text_word_predictions
                )

            if level == "entity":
                assert (
                    proba is False
                ), f"ERROR! level = entity not allowed if proba = {proba}"
                input_text_word_predictions = merge_tokens_to_entities(
                    input_text_word_predictions, input_text
                )

            predictions.append(input_text_word_predictions)

        ######################################
        # all input_texts, merged chunks, words
        ######################################
        return predictions

    ####################################################################################################################
    # PREDICT HELPER METHODS
    ####################################################################################################################
    def _get_predict_dataloader(self, examples):
        """
        :param examples:            [list] of [str]
        :return: predict_dataloader [torch dataloader]
        """
        # input_examples
        input_examples = self.data_preprocessor.get_input_examples_predict(
            examples=examples,
        )

        # dataloader
        dataloader = self.data_preprocessor.to_dataloader(
            input_examples, self.tag_list, batch_size=1
        )

        return dataloader["predict"]

    def _predict_on_tokens(self, sample):
        """
        :param sample: [list] w/ 4 tensors: input_ids, attention_mask, segment_ids, label_ids
        :return: output_token_tensors [list] of [torch tensor] of shape [1, #tags]
        :return: tokens               [list] of [str]
        """
        (
            input_ids,  # shape: [1, seq_length]
            attention_mask,  # shape: [1, seq_length]
            segment_ids,  # shape: [1, seq_length]
            label_ids,  # shape: [1, seq_length]
        ) = sample

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
        )  # shape: [1 (=#examples), 1 (=#batch_size), seq_length, #tags]

        output_token_tensors = [
            output[0][0][i]  # .detach().numpy()
            for i in range(self._hparams.max_seq_length)
        ]  # shape: [seq_length, #tags]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return output_token_tensors, tokens

    def _turn_tensors_into_tags(self, output_token_tensors):
        """
        :param output_token_tensors       [list] of [torch tensor] of shape [1, #tags]
        :return: output_token_predictions [list] of [str]
        """
        return [
            self.tag_list[np.argmax(output_token_tensors[i].detach().numpy())]
            for i in range(self._hparams.max_seq_length)
        ]

    def _turn_tensors_into_tag_probability_distributions(self, output_token_tensors):
        """
        :param output_token_tensors       [list] of [torch tensor] of shape [1, #tags]
        :return: output_token_predictions [list] of [prob dist], i.e. dict that maps tags to probabilities
        """

        probability_distributions = [
            softmax(output_token_tensors[i], dim=0)
            for i in range(self._hparams.max_seq_length)
        ]

        tag_probability_distribution = [
            {
                self.tag_list[j]: float(
                    probability_distributions[i][j].detach().numpy()
                )
                for j in range(len(self.tag_list))
            }
            for i in range(self._hparams.max_seq_length)
        ]

        return tag_probability_distribution


########################################################################################################################
########################################################################################################################
########################################################################################################################
def get_tags_on_words_between_special_tokens(
    tokens: List[str], example_token_predictions: List[Any]
) -> List[Tuple[Union[str, Any], ...]]:
    """
    Args:
        tokens:                     [list] of [str]
        example_token_predictions:  [list] of [str] or [prob dist]

    Returns:
        example_word_predictions:   [list] of Tuple[str, str or prob dist]
    """
    # predict tags on words between [CLS] and [SEP]
    example_word_predictions_list: List[List[Union[str, Any]]] = list()
    for token, example_token_prediction in zip(tokens, example_token_predictions):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            if not token.startswith("##"):
                example_word_predictions_list.append([token, example_token_prediction])
            else:
                example_word_predictions_list[-1][0] += token.strip("##")

    example_word_predictions = [tuple(elem) for elem in example_word_predictions_list]

    return example_word_predictions


def restore_unknown_tokens(
    example_word_predictions: List[Tuple[Union[str, Any], ...]],
    _example: str,
    verbose: bool = True,
) -> List[Dict[str, Union[str, Dict]]]:
    """
    - replace "[UNK]" tokens by the original token
    - enrich tokens with char_start & char_end

    Args:
        example_word_predictions: e.g. [('example', 'O'), ('of', 'O), ('author', 'PERSON'), ..]
        _example: 'example of author'
        verbose:

    Returns:
        example_word_predictions_restored: e.g. [
            {"char_start": "0", "char_end": "7", "token": "example", "tag": "O"},
            ..
        ]
    """
    example_word_predictions_restored = list()

    # 1. get margins of known tokens
    token_char_margins: List[Tuple[Any, ...]] = list()
    char_start = 0
    for token, _ in example_word_predictions:
        if token != "[UNK]":
            try:
                char_start = _example.index(token, char_start)
                token_char_margins.append((char_start, char_start + len(token)))
                char_start += len(token)
            except ValueError:
                print(f"! token = {token} not found in example[{char_start}:]")
                token_char_margins.append((None, None))
        else:
            token_char_margins.append((None, None))
            char_start_new = None
            for expression in [" ", "."]:  # TODO: heuristic, could/should be improved
                try:
                    char_start_new = _example.index(expression, char_start + 1)
                except ValueError:
                    pass
            if char_start_new is not None:
                char_start = max(char_start, char_start_new)

    # 2. restore unknown tokens
    for i, (token, tag) in enumerate(example_word_predictions):
        if (
            token_char_margins[i][0] is not None
            and token_char_margins[i][1] is not None
        ):
            example_word_predictions_restored.append(
                {
                    "char_start": str(token_char_margins[i][0]),
                    "char_end": str(token_char_margins[i][1]),
                    "token": token,
                    "tag": tag,
                }
            )
        else:
            for j in range(2):
                assert (
                    token_char_margins[i][j] is None
                ), f"ERROR! token_char_margin[{i}][{j}] is not None for token = {token}"

            char_start_margin, char_end_margin = None, None
            for k in range(1, 10):
                if i - k < 0:
                    char_start_margin = 0
                    break
                elif token_char_margins[i - k][1] is not None:
                    char_start_margin = token_char_margins[i - k][1]
                    break
            for k in range(1, 10):
                if i + k >= len(token_char_margins):
                    char_end_margin = len(_example)
                    break
                elif token_char_margins[i + k][0] is not None:
                    char_end_margin = token_char_margins[i + k][0]
                    break
            assert (
                char_start_margin is not None
            ), f"ERROR! could not find char_start_margin"
            assert char_end_margin is not None, f"ERROR! could not find char_end_margin"

            new_token = _example[char_start_margin:char_end_margin].strip()

            if len(new_token):
                char_start = _example.index(new_token, char_start_margin)
                char_end = char_start + len(new_token)
                if verbose:
                    print(
                        f"! restored unknown token = {new_token} between "
                        f"char_start = {char_start_margin}, "
                        f"char_end = {char_end_margin}"
                    )
                example_word_predictions_restored.append(
                    {
                        "char_start": str(char_start),
                        "char_end": str(char_end),
                        "token": new_token,
                        "tag": tag,
                    }
                )
            else:
                print(
                    f"! dropped unknown empty token between "
                    f"char_start = {char_start_margin}, "
                    f"char_end = {char_end_margin}"
                )

    return example_word_predictions_restored


def restore_annotation_scheme_consistency(
    example_word_predictions: List[Dict[str, Union[str, Dict]]],
) -> List[Dict[str, Union[str, Dict]]]:
    """
    restore annotation scheme consistency in case of BIO tags
    plain tags are not modified

    Args:
        example_word_predictions: e.g. [
            {"char_start": "0", "char_end": "7", "token": "example", "tag": "I-TAG"},
            ..
        ]

    Returns:
        example_word_predictions_restored: e.g. [
            {"char_start": "0", "char_end": "7", "token": "example", "tag": "B-TAG"},
            ..
        ]
    """
    example_word_predictions_restored: List[Dict[str, Union[str, Dict]]] = list()
    for i in range(len(example_word_predictions)):
        example_word_prediction_restored = example_word_predictions[i]
        current_tag = example_word_predictions[i]["tag"]
        current_tag = assert_str(current_tag, "current_tag")

        if current_tag == "O" or "-" not in current_tag or current_tag.startswith("B-"):
            example_word_predictions_restored.append(example_word_prediction_restored)
        elif current_tag.startswith("I-"):
            previous_tag = example_word_predictions[i - 1]["tag"] if i > 0 else None

            if previous_tag not in [current_tag, current_tag.replace("I-", "B-")]:
                example_word_prediction_restored["tag"] = current_tag.replace(
                    "I-", "B-"
                )

            example_word_predictions_restored.append(example_word_prediction_restored)
        else:
            raise Exception(
                f"ERROR! current tag = {current_tag} expected to be of the form I-*"
            )

    assert len(example_word_predictions_restored) == len(
        example_word_predictions
    ), f"ERROR!"

    return example_word_predictions_restored


def merge_tokens_to_entities(
    example_word_predictions: List[Dict[str, Union[str, Dict]]],
    example: str,
) -> List[Dict[str, Union[str, Dict]]]:
    """
    merge token predictions that belong together (B-* & I-*)
    discard tokens with tag 'O'

    Args:
        example_word_predictions: e.g. [
            {"char_start": "0", "char_end": "7", "token": "example", "tag": "B-TAG"},
            {"char_start": "8", "char_end": "16", "token": "sentence", "tag": "I-TAG"},
            {"char_start": "17", "char_end": "18", "token": ".", "tag": "O"},
            ..
        ]
        example: str

    Returns:
        example_word_predictions_merged: e.g. [
            {"char_start": "0", "char_end": "16", "token": "example", "tag": "TAG"},
            ..
        ]
    """
    merged_ner_tags = list()
    count = {
        "o_tags": 0,
        "replace": 0,
        "merge": 0,
        "unmodified": 0,
    }
    for i in range(len(example_word_predictions)):
        current_tag = example_word_predictions[i]["tag"]
        current_tag = assert_str(current_tag, "current_tag")
        if current_tag == "O":
            # merged_ner_tag = example_word_predictions[i]
            count["o_tags"] += 1
            # merged_ner_tags.append(merged_ner_tag)
        elif current_tag.startswith("B-"):  # BIO scheme
            n_tags = 0
            for n in range(i + 1, len(example_word_predictions)):
                next_tag = example_word_predictions[n]["tag"]
                next_tag = assert_str(next_tag, "next_tag")
                # next_token_start = example_word_predictions[n].token_start
                # previous_token_end = example_word_predictions[n - 1].token_end
                if next_tag.startswith("I-") and next_tag == current_tag.replace(
                    "B-", "I-"
                ):  # and previous_token_end == next_token_start:
                    n_tags += 1
                else:
                    break

            merged_ner_tag = example_word_predictions[i]
            assert isinstance(
                merged_ner_tag["tag"], str
            ), "ERROR! merged_ner_tag.tag should be a string"

            merged_ner_tag["tag"] = merged_ner_tag["tag"].split("-")[-1]
            if n_tags > 0:
                merged_ner_tag["char_end"] = example_word_predictions[i + n_tags][
                    "char_end"
                ]
                # merged_ner_tag.token_end = example_word_predictions[i + n_tags]["token_end"]
                assert isinstance(
                    merged_ner_tag["char_start"], str
                ), "ERROR! merged_ner_tag.char_start should be a string"
                assert isinstance(
                    merged_ner_tag["char_end"], str
                ), "ERROR! merged_ner_tag.char_end should be a string"
                merged_ner_tag["token"] = example[
                    int(merged_ner_tag["char_start"]) : int(merged_ner_tag["char_end"])
                ]
                count["merge"] += 1 + n_tags
            else:
                count["replace"] += 1
            merged_ner_tags.append(merged_ner_tag)
        elif "-" not in current_tag:  # plain scheme
            count["unmodified"] += 1
            merged_ner_tag = example_word_predictions[i]
            merged_ner_tags.append(merged_ner_tag)

    assert count["merge"] + count["replace"] + count["o_tags"] + count[
        "unmodified"
    ] == len(
        example_word_predictions
    ), f"{count} -> {sum(count.values())} != {len(example_word_predictions)} | {example_word_predictions}"

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

    example_word_predictions_merged = merged_ner_tags
    if VERBOSE:
        print(
            f"> merged {len(example_word_predictions_merged)} BIO-tags "
            f"(simple replace: {count['replace']}, merge: {count['merge']}, O-tags: {count['o_tags']}, unmodified: {count['unmodified']}).\n"
        )

    return example_word_predictions_merged


def assert_str(_object: Union[str, Dict[Any, Any]], _object_name: str) -> str:
    assert isinstance(
        _object, str
    ), f"ERROR! {_object_name} = {_object} is {type(_object)} but should be string"
    return _object

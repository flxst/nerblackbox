import json
from os.path import join, isdir
from typing import Tuple, Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import List, Union, Dict
from torch.nn.functional import softmax
from nerblackbox.modules.ner_training.annotation_tags.token_tags import TokenTags
from nerblackbox.modules.ner_training.data_preprocessing.data_preprocessor import DataPreprocessor
from nerblackbox.tests.utils import PseudoDefaultLogger

VERBOSE = False

PREDICTIONS = List[List[Dict[str, Union[str, Dict]]]]


class NerModelPredict:
    """
    class that predicts tags for given text
    """
    @classmethod
    def checkpoint_exists(cls, checkpoint_directory: str) -> bool:
        return isdir(checkpoint_directory)

    def __init__(self, checkpoint_directory: str, batch_size: int = 16, max_seq_length: int = None):
        """
        Args:
            checkpoint_directory
            batch_size: used in dataloader
        """
        # 0. device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1: max_seq_length
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        else:
            path_max_seq_length = join(checkpoint_directory, "max_seq_length.json")
            with open(path_max_seq_length, "r") as f:
                self.max_seq_length = json.load(f)

        # 2. annotation
        path_annotation_classes = join(checkpoint_directory, "annotation_classes.json")
        with open(path_annotation_classes, "r") as f:
            self.annotation_classes = json.load(f)
        id2label = {i: label for i, label in enumerate(self.annotation_classes)}
        label2id = {label: i for i, label in id2label.items()}

        self.annotation_scheme = self._derive_annotation_scheme(id2label)

        # 3. model
        self.model = AutoModelForTokenClassification.from_pretrained(
            checkpoint_directory,
            id2label=id2label,
            label2id=label2id,
            return_dict=False,
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        # 4. tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_directory,
        )

        # 5. batch_size (dataloader)
        self.batch_size = batch_size

    def predict(
            self,
            input_texts: Union[str, List[str]],
            level: str = "entity",
            autocorrect: bool = False,
    ) -> PREDICTIONS:
        """predict tags for input texts. output on entity or word level.

        Examples:
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="word", autocorrect=False)
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "I-ORG"},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
            #     {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-LOC"},
            # ]]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="word", autocorrect=True)
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "B-ORG"},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "tag": "O"},
            #     {"char_start": "25", "char_end": "26", "token": "i", "tag": "O"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "B-LOC"},
            # ]]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="entity", autocorrect=False)
            # [[
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "LOC"},
            # ]]
            ```
            ```
            predict(["arbetsförmedlingen finns i stockholm"], level="entity", autocorrect=True)
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "tag": "ORG"},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "tag": "LOC"},
            # ]]
            ```

        Args:
            input_texts:   e.g. ["example 1", "example 2"]
            level:         "entity" or "word"
            autocorrect:   if True, autocorrect annotation scheme (e.g. B- and I- tags).

        Returns:
            predictions: [list] of predictions for the different examples.
                         each list contains a [list] of [dict] w/ keys = char_start, char_end, word, tag
        """
        return self._predict(input_texts, level, autocorrect, proba=False)

    def predict_proba(self, input_texts: Union[str, List[str]]) -> PREDICTIONS:
        """predict probability distributions for input texts. output on word level.

        Examples:
            ```
            predict_proba(["arbetsförmedlingen finns i stockholm"])
            # [[
            #     {"char_start": "0", "char_end": "18", "token": "arbetsförmedlingen", "proba_dist: {"O": 0.21, "B-ORG": 0.56, ..}},
            #     {"char_start": "19", "char_end": "24", "token": "finns", "proba_dist: {"O": 0.87, "B-ORG": 0.02, ..}},
            #     {"char_start": "25", "char_end": "26", "token": "i", "proba_dist: {"O": 0.95, "B-ORG": 0.01, ..}},
            #     {"char_start": "27", "char_end": "36", "token": "stockholm", "proba_dist: {"O": 0.14, "B-ORG": 0.22, ..}},
            # ]]
            ```

        Args:
            input_texts:   e.g. ["example 1", "example 2"]

        Returns:
            predictions: [list] of probability predictions for different examples.
                         each list contains a [list] of [dict] w/ keys = char_start, char_end, word, proba_dist
                         where proba_dist = [dict] that maps self.annotation.classes to probabilities
        """
        return self._predict(input_texts, level="word", autocorrect=False, proba=True)

    def _predict(
            self,
            input_texts: Union[str, List[str]],
            level: str = "entity",
            autocorrect: bool = False,
            proba: bool = False,
    ) -> PREDICTIONS:
        """predict tags or probabilities for tags

        Args:
            input_texts:  e.g. ["example 1", "example 2"]
            level:        "entity" or "word"
            autocorrect:  if True, autocorrect annotation scheme (e.g. B- and I- tags).
            proba:        if True, predict probabilities instead of labels (on word level)

        Returns:
            predictions: [list] of [list] of [dict] w/ keys = char_start, char_end, word, tag/proba_dist
                         where proba_dist = [dict] that maps self.annotation.classes to probabilities
        """
        # --- check input arguments ---
        assert level in [
            "entity",
            "word",
        ], f"ERROR! model prediction level = {level} unknown, needs to be entity or word."
        if proba:
            assert level == "word" and autocorrect is False, (
                f"ERROR! probability predictions require level = word and autocorrect = False. "
                f"level = {level} and autocorrect = {autocorrect} not possible."
            )
        # ------------------------------

        # 0. ensure input_texts is a list
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        number_of_input_texts = len(input_texts)

        # 1. pure model inference
        self.data_preprocessor = DataPreprocessor(
            tokenizer=self.tokenizer,
            do_lower_case=False,
            max_seq_length=self.max_seq_length,
            default_logger=PseudoDefaultLogger(),
        )
        input_examples = self.data_preprocessor.get_input_examples_predict(
            examples=input_texts,
        )
        dataloader, offsets = self.data_preprocessor.to_dataloader(
            input_examples, self.annotation_classes, batch_size=self.batch_size
        )
        dataloader = dataloader['predict']
        offsets = offsets['predict']

        inputs_batches = list()
        outputs_batches = list()
        for sample in dataloader:
            inputs_batches.append(sample['input_ids'])
            sample.pop("labels")
            sample = {k: v.to(self.device) for k, v in sample.items()}
            with torch.no_grad():
                outputs_batch = self.model(**sample)[0]  # shape = [batch_size, seq_length, num_labels]
            outputs_batches.append(outputs_batch.detach().cpu())

        inputs = torch.cat(inputs_batches)  # shape = [number_of_input_texts, seq_length]
        outputs = torch.cat(outputs_batches)  # shape = [number_of_input_texts, seq_length, num_labels]

        ################################################################################################################
        ################################################################################################################
        tokens = [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs.tolist()]

        if proba:
            predictions = self._turn_tensors_into_tag_probability_distributions(output_token_tensors=outputs)
        else:
            predictions_ids = torch.argmax(outputs, dim=2)
            predictions = [
                [self.model.config.id2label[prediction] for prediction in predictions_ids[i].numpy()]
                for i in range(len(predictions_ids))
            ]

        # merge
        tokens = [
            self._merge_slices_for_single_document(tokens[offsets[i]:offsets[i+1]])
            for i in range(len(offsets)-1)
        ]
        predictions = [
            self._merge_slices_for_single_document(predictions[offsets[i]:offsets[i+1]])
            for i in range(len(offsets)-1)
        ]

        assert len(tokens) == len(predictions), \
            f"ERROR! len(tokens) = {len(tokens)} should equal len(predictions) = {len(predictions)}"
        assert len(tokens) == number_of_input_texts, \
            f"ERROR! len(tokens) = {len(tokens)} should equal len(input_texts) = {number_of_input_texts}"
        assert len(predictions) == number_of_input_texts, \
            f"ERROR! len(predictions) = {len(predictions)} should equal len(input_texts) = {number_of_input_texts}"

        # 2. post processing
        predictions = [
            self._post_processing(level, autocorrect, proba, input_texts[i], tokens[i], predictions[i])
            for i in range(number_of_input_texts)
        ]

        return predictions

    @staticmethod
    def _merge_slices_for_single_document(_list: List[Union[Dict, List[Dict]]]) -> List[Dict]:
        """
        merges the slices for a single document

        Args:
            _list: predictions for one or more slices that belong to same document,
                   e.g. one slice:  [{"char_start": ..}, ..],
                   e.g. two slices: [[{"char_start": ..}, ..], [{"char_start": ..}, ..]],

        Returns:
            _list_flat: predictions for one document
        """
        if len(_list) == 1:  # one slice
            return _list[0]
        else:  # more slices
            _list_flat = list()
            for i, sublist in enumerate(_list):
                if i == 0:
                    _list_flat.extend(sublist[:-1])
                elif 0 < i < len(_list) - 1:
                    _list_flat.extend(sublist)
                else:
                    _list_flat.extend(sublist[1:])
            return _list_flat

    def _post_processing(self, level, autocorrect, proba, input_text, input_text_tokens, input_text_token_predictions):
        """

        :param level:
        :param autocorrect:
        :param proba:
        :param input_text:
        :param input_text_tokens:
        :param input_text_token_predictions:
        :return:
        """
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

        if autocorrect or level == "entity":
            assert (
                    proba is False
            ), f"ERROR! autocorrect = {autocorrect} / level = {level} not allowed if proba = {proba}"

            def assert_typing(input_text_word_predictions_str_dict):
                # this is only to ensure correct typing, it does not actually change anything
                return [
                    {
                        k: str(v)
                        for k, v in input_text_word_prediction_str_dict.items()
                    }
                    for input_text_word_prediction_str_dict in input_text_word_predictions_str_dict
                ]

            input_text_word_predictions_str: List[Dict[str, str]] = assert_typing(
                input_text_word_predictions
            )

            token_tags = TokenTags(
                input_text_word_predictions_str, scheme=self.annotation_scheme
            )

            if autocorrect:
                token_tags.restore_annotation_scheme_consistency()

            if level == "entity":
                token_tags.merge_tokens_to_entities(
                    original_text=input_text, verbose=VERBOSE
                )

            input_text_word_predictions = token_tags.as_list()

        return input_text_word_predictions

    @staticmethod
    def _derive_annotation_scheme(_id2label):
        label_prefixes = list(set([label.split("-")[0] for label in _id2label.values() if "-" in label]))
        if len(label_prefixes) == 0:
            return "plain"
        elif "B" in label_prefixes and "I" in label_prefixes and "L" in label_prefixes and "U" in label_prefixes:
            return "bilou"
        elif "B" in label_prefixes and "I" in label_prefixes:
            return "bio"
        else:
            raise Exception(f"ERROR! could not derive annotation scheme. found label_prefixes = {label_prefixes}")

    def _turn_tensors_into_tag_probability_distributions(self, output_token_tensors):
        """
        :param output_token_tensors       [torch tensor] of shape [batch_size, seq_length, #tags]
        :return: output_token_predictions [list] of [list] of [prob dist], i.e. dict that maps tags to probabilities
        """
        probability_distributions = softmax(output_token_tensors, dim=2)
        tag_probability_distribution = [
            [
                {
                    self.annotation_classes[j]: float(
                        probability_distributions[x][i][j].detach().numpy()
                    )
                    for j in range(len(self.annotation_classes))
                }
                for i in range(output_token_tensors.shape[1])
            ]
            for x in range(output_token_tensors.shape[0])
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

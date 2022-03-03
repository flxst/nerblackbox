from copy import deepcopy
from typing import List, Tuple, Dict, Union, Optional

Encodings = List[Tuple[int, str, str]]
Predictions = List[Dict[str, Union[str, Dict]]]


class TextEncoder:
    def __init__(
        self,
        encodings_mapping: Dict[str, str],
        model_special_tokens: Optional[List[str]] = None,
    ):
        r"""
        Args:
            encodings_mapping: e.g. {"\n": "[NEWLINE]", "\t": "[TAB]"}
            model_special_tokens: e.g. ["[NEWLINE]", "[TAB]"]
        """
        self.encodings_mapping = encodings_mapping
        if model_special_tokens is None:
            print("ATTENTION! DID NOT CHECK THAT MODEL WAS TRAINED WITH SPECIAL TOKENS ACCORDING TO encodings_mapping! "
                  "It is recommended to provide an 'model_special_tokens' argument to check.")
        else:
            assert sorted(list(set(encodings_mapping.values()))) == sorted(list(set(model_special_tokens))), \
                f"ERROR! encodings_mapping values = {sorted(list(set(encodings_mapping.values())))} does not equal " \
                f"model_special_tokens = {sorted(list(set(model_special_tokens)))}"

    def encode(self, text_list: List[str]) -> Tuple[List[str], List[Encodings]]:
        r"""
        encodes list of text using self.encodings_mapping

        Args:
            text_list: e.g. ["an\n example"]

        Returns:
            text_encoded_list: e.g. ["an[NEWLINE] example"]
            encodings_list: e.g. [[(2, "\n", "[NEWLINE]")]]
        """
        list_of_single_encodings = [
            self._encode_single(text)
            for text in text_list
        ]
        return [elem[0] for elem in list_of_single_encodings], [elem[1] for elem in list_of_single_encodings]

    def decode(self,
               text_encoded_list: List[str],
               encodings_list: List[Encodings],
               predictions_encoded_list: List[Predictions]) -> Tuple[List[str], List[Predictions]]:
        r"""
        decodes list of text_encoded and predictions_encoded using encodings

        Args:
            text_encoded_list: e.g. ["an[NEWLINE] example"]
            encodings_list: e.g. [[(2, "\n", "[NEWLINE]")]]
            predictions_encoded_list: e.g. [[{"char_start": "12", "char_end": "19", "token": "example", "tag": "TAG"}]]

        Returns:
            text_list: e.g. ["an\n example"]
            predictions_list: e.g. [[{"char_start": "4", "char_end": "11", "token": "example", "tag": "TAG"}]]
        """
        list_of_single_decodings = [
            self._decode_single(text_encoded, encodings, predictions_encoded)
            for text_encoded, encodings, predictions_encoded
            in zip(text_encoded_list, encodings_list, predictions_encoded_list)
        ]
        return [elem[0] for elem in list_of_single_decodings], [elem[1] for elem in list_of_single_decodings]

    def _encode_single(self, text: str) -> Tuple[str, Encodings]:
        r"""
        encodes single text using self.encodings_mapping

        Args:
            text: e.g. "an\n example"

        Returns:
            text_encoded: e.g. "an[NEWLINE] example"
            encodings: e.g. [(2, "\n", "[NEWLINE]")]
        """
        text_encoded = deepcopy(text)
        encodings = list()
        for k, v in self.encodings_mapping.items():
            while k in text_encoded:
                index = text_encoded.find(k)
                text_encoded = text_encoded.replace(k, v, 1)
                encodings.append((index, k, v))
        encodings.reverse()
        return text_encoded, encodings

    @staticmethod
    def _decode_single(text_encoded: str,
                       encodings: Encodings,
                       predictions_encoded: Predictions) -> Tuple[str, Predictions]:
        r"""
        decodes single text_encoded and predictions_encoded using encodings

        Args:
            text_encoded: e.g. "an[NEWLINE] example"
            encodings: e.g. [(2, "\n", "[NEWLINE]")]
            predictions_encoded: e.g. [{"char_start": "12", "char_end": "19", "token": "example", "tag": "TAG"}]

        Returns:
            text: e.g. "an\n example"
            predictions: e.g. [{"char_start": "4", "char_end": "11", "token": "example", "tag": "TAG"}]
        """
        text = deepcopy(text_encoded)
        predictions = deepcopy(predictions_encoded)
        for encoding in encodings:
            index, k, v = encoding
            # print(index, k, v)
            assert text[index:index + len(v)] == v, \
                f"ERROR! text[{index}:{index + len(v)}] = {text[index:index + len(v)]} != {v}"
            text = text[:index] + k + text[index + len(v):]

            for prediction in predictions:
                if int(prediction['char_end']) > index:
                    prediction['char_start'] = str(int(prediction['char_start']) - len(v) + len(k))
                    prediction['char_end'] = str(int(prediction['char_end']) - len(v) + len(k))

        return text, predictions

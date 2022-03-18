from copy import deepcopy
from typing import List, Tuple, Dict, Union, Optional

EncodeDecodeMappings = List[Tuple[int, str, str]]
Predictions = List[Dict[str, Union[str, Dict]]]


class TextEncoder:
    def __init__(
        self,
        encoding: Dict[str, str],
        model_special_tokens: Optional[List[str]] = None,
    ):
        r"""
        Args:
            encoding: e.g. {"\n": "[NEWLINE]", "\t": "[TAB]"}
            model_special_tokens: e.g. ["[NEWLINE]", "[TAB]"]
        """
        self.encoding = encoding
        if model_special_tokens is None:
            print("ATTENTION! DID NOT CHECK THAT MODEL WAS TRAINED WITH SPECIAL TOKENS ACCORDING TO encoding! "
                  "It is recommended to provide an 'model_special_tokens' argument to check.")
        else:
            assert sorted(list(set(encoding.values()))) == sorted(list(set(model_special_tokens))), \
                f"ERROR! encoding values = {sorted(list(set(encoding.values())))} does not equal " \
                f"model_special_tokens = {sorted(list(set(model_special_tokens)))}"

    def encode(self, text_list: List[str]) -> Tuple[List[str], List[EncodeDecodeMappings]]:
        r"""
        encodes list of text using self.encoding

        Args:
            text_list: e.g. ["an\n example"]

        Returns:
            text_encoded_list: e.g. ["an[NEWLINE] example"]
            encode_decode_mappings_list: e.g. [[(2, "\n", "[NEWLINE]")]]
        """
        list_of_single_encodings = [
            self._encode_single(text)
            for text in text_list
        ]
        return [elem[0] for elem in list_of_single_encodings], [elem[1] for elem in list_of_single_encodings]

    def decode(self,
               text_encoded_list: List[str],
               encode_decode_mappings_list: List[EncodeDecodeMappings],
               predictions_encoded_list: List[Predictions]) -> Tuple[List[str], List[Predictions]]:
        r"""
        decodes list of text_encoded and predictions_encoded using encode_decode_mappings

        Args:
            text_encoded_list: e.g. ["an[NEWLINE] example"]
            encode_decode_mappings_list: e.g. [[(2, "\n", "[NEWLINE]")]]
            predictions_encoded_list: e.g. [[{"char_start": "12", "char_end": "19", "token": "example", "tag": "TAG"}]]

        Returns:
            text_list: e.g. ["an\n example"]
            predictions_list: e.g. [[{"char_start": "4", "char_end": "11", "token": "example", "tag": "TAG"}]]
        """
        list_of_single_decodings = [
            self._decode_single(text_encoded, encode_decode_mappings, predictions_encoded)
            for text_encoded, encode_decode_mappings, predictions_encoded
            in zip(text_encoded_list, encode_decode_mappings_list, predictions_encoded_list)
        ]
        return [elem[0] for elem in list_of_single_decodings], [elem[1] for elem in list_of_single_decodings]

    def _encode_single(self, text: str) -> Tuple[str, EncodeDecodeMappings]:
        r"""
        encodes single text using self.encoding

        Args:
            text: e.g. "an\n example"

        Returns:
            text_encoded: e.g. "an[NEWLINE] example"
            encode_decode_mappings: e.g. [(2, "\n", "[NEWLINE]")]
        """
        text_encoded = deepcopy(text)
        encode_decode_mappings = list()
        for k, v in self.encoding.items():
            while k in text_encoded:
                index = text_encoded.find(k)
                text_encoded = text_encoded.replace(k, v, 1)
                encode_decode_mappings.append((index, k, v))
        encode_decode_mappings.reverse()
        return text_encoded, encode_decode_mappings

    @staticmethod
    def _decode_single(text_encoded: str,
                       encode_decode_mappings: EncodeDecodeMappings,
                       predictions_encoded: Predictions) -> Tuple[str, Predictions]:
        r"""
        decodes single text_encoded and predictions_encoded using encode_decode_mapping

        Args:
            text_encoded: e.g. "an[NEWLINE] example"
            encode_decode_mappings: e.g. [(2, "\n", "[NEWLINE]")]
            predictions_encoded: e.g. [{"char_start": "12", "char_end": "19", "token": "example", "tag": "TAG"}]

        Returns:
            text: e.g. "an\n example"
            predictions: e.g. [{"char_start": "4", "char_end": "11", "token": "example", "tag": "TAG"}]
        """
        text = deepcopy(text_encoded)
        predictions = deepcopy(predictions_encoded)
        for encode_decode_mapping in encode_decode_mappings:
            index, k, v = encode_decode_mapping
            # print(index, k, v)
            assert text[index:index + len(v)] == v, \
                f"ERROR! text[{index}:{index + len(v)}] = {text[index:index + len(v)]} != {v}"
            text = text[:index] + k + text[index + len(v):]

            for prediction in predictions:
                assert isinstance(prediction['char_start'], str) and isinstance(prediction['char_end'], str), \
                    f"ERROR! expected str, got type ({prediction['char_start']}, {prediction['char_end']})"
                if int(prediction['char_start']) == index and int(prediction['char_end']) == index + len(v):
                    prediction['char_end'] = str(int(prediction['char_end']) - len(v) + len(k))
                    prediction['token'] = k
                elif int(prediction['char_end']) > index:
                    prediction['char_start'] = str(int(prediction['char_start']) - len(v) + len(k))
                    prediction['char_end'] = str(int(prediction['char_end']) - len(v) + len(k))

        return text, predictions

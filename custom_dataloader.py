import numpy as np
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from dataclasses import dataclass
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)


@dataclass
class CustomWithNegativeDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        inputs = (
            [np.array(feature["input_ids"]) for feature in features]
            if "input_ids" in features[0].keys()
            else None
        )
        if "synonym_inputs" in features[0].keys():
            synonym_inputs = [
                np.array(feature["synonym_inputs"]) for feature in features
            ]
        else:
            None
        if "random_inputs" in features[0].keys():
            random_inputs = [np.array(feature["random_inputs"]) for feature in features]
        else:
            None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        new_labels = [feature["labels"] for feature in features]

        if (
            "synonym_inputs" in features[0].keys()
            and "random_inputs" not in features[0].keys()
        ):
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs + synonym_inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        elif (
            "synonym_inputs" not in features[0].keys()
            and "random_inputs" in features[0].keys()
        ):
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs + random_inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        elif (
            "synonym_inputs" in features[0].keys()
            and "random_inputs" in features[0].keys()
        ):
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs + synonym_inputs + random_inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        else:
            stack_features = self.tokenizer.pad(
                {"input_ids": inputs},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

        if (
            "synonym_inputs" in features[0].keys()
            and "random_inputs" not in features[0].keys()
        ):
            stack_features["labels"] = self.tokenizer.pad(
                {"input_ids": new_labels + new_labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )["input_ids"]
        elif (
            "synonym_inputs" not in features[0].keys()
            and "random_inputs" in features[0].keys()
        ):
            stack_features["labels"] = self.tokenizer.pad(
                {"input_ids": new_labels + new_labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )["input_ids"]
        elif (
            "synonym_inputs" in features[0].keys()
            and "random_inputs" in features[0].keys()
        ):
            stack_features["labels"] = self.tokenizer.pad(
                {"input_ids": new_labels + new_labels + new_labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )["input_ids"]
        else:
            stack_features["labels"] = self.tokenizer.pad(
                {"input_ids": new_labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )["input_ids"]

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=stack_features["labels"]
            )
            stack_features["decoder_input_ids"] = decoder_input_ids

        return stack_features


@dataclass
class CustomDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        inputs = [np.array(feature["input_ids"]) for feature in features]

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        new_labels = [feature["labels"] for feature in features]

        stack_features = self.tokenizer.pad(
            {"input_ids": inputs},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        stack_features["labels"] = self.tokenizer.pad(
            {"input_ids": new_labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )["input_ids"]

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=stack_features["labels"]
            )
            stack_features["decoder_input_ids"] = decoder_input_ids

        return stack_features

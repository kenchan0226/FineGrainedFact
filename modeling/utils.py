# Copyright (c) 2020, Salesforce.com, Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import json
import os
import sys
from io import open
from datasets import load_metric
import torch

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, recall_score, precision_score, roc_auc_score

from collections import Counter

logger = logging.getLogger(__name__)
seq_label_metric = load_metric("seqeval")


class FactCCGeneratedAttnDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        #self.files = os.listdir(root) # take all files in the root directory
        num_samples = len(os.listdir(root))
        self.files = ["{}.pt".format(i) for i in range(num_samples)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_ids, input_mask, segment_ids, label_ids, doc_frames_word_mask, claim_frames_word_mask, \
        claim_attn_mask, claim_frames_padding_mask, doc_words_mask, doc_frames_np_word_mask, doc_frames_predicate_word_mask, claim_frames_np_word_mask, claim_frames_predicate_word_mask, guid = torch.load(os.path.join(self.root, self.files[idx])) # load the features of this sample
        return input_ids, input_mask, segment_ids, label_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask, claim_frames_padding_mask, doc_words_mask, doc_frames_np_word_mask, doc_frames_predicate_word_mask, claim_frames_np_word_mask, claim_frames_predicate_word_mask, guid


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 extraction_span=None, augmentation_span=None, original_text_b=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.extraction_span = extraction_span
        self.augmentation_span = augmentation_span
        self.original_text_b = original_text_b


class InputAttnExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 doc_frames_word_ids=None, claim_frames_word_ids=None, original_text_b=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        #self.extraction_span = extraction_span
        self.doc_frames_word_ids = doc_frames_word_ids
        self.claim_frames_word_ids = claim_frames_word_ids
        self.original_text_b = original_text_b


class InputSeqLabelExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 extraction_span=None, augmentation_seq_label=None, original_text_b=None, augmentation_seq_label_mask=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.extraction_span = extraction_span
        self.original_text_b = original_text_b
        self.augmentation_seq_label = augmentation_seq_label
        self.augmentation_seq_label_mask = augmentation_seq_label_mask


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 extraction_mask=None, extraction_start_ids=None, extraction_end_ids=None,
                 augmentation_mask=None, augmentation_start_ids=None, augmentation_end_ids=None, guid=None, original_text_b_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.extraction_mask = extraction_mask
        self.extraction_start_ids = extraction_start_ids
        self.extraction_end_ids = extraction_end_ids
        self.augmentation_mask = augmentation_mask
        self.augmentation_start_ids = augmentation_start_ids
        self.augmentation_end_ids = augmentation_end_ids
        self.guid = guid
        self.original_text_b_ids = original_text_b_ids


class InputAttnFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 doc_frames_word_mask=None, doc_frames_np_word_mask=None, doc_frames_predicate_word_mask=None, claim_frames_word_mask=None, claim_frames_np_word_mask=None, claim_frames_predicate_word_mask=None, claim_attn_mask=None, claim_frames_padding_mask=None, doc_words_mask=None, guid=None, original_text_b_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        """
        self.extraction_mask = extraction_mask
        self.extraction_start_ids = extraction_start_ids
        self.extraction_end_ids = extraction_end_ids
        self.augmentation_mask = augmentation_mask
        self.augmentation_start_ids = augmentation_start_ids
        self.augmentation_end_ids = augmentation_end_ids
        """
        self.doc_frames_word_mask = doc_frames_word_mask
        self.doc_frames_np_word_mask = doc_frames_np_word_mask
        self.doc_frames_predicate_word_mask = doc_frames_predicate_word_mask
        self.claim_frames_word_mask = claim_frames_word_mask
        self.claim_frames_np_word_mask = claim_frames_np_word_mask
        self.claim_frames_predicate_word_mask = claim_frames_predicate_word_mask
        self.claim_attn_mask = claim_attn_mask
        self.claim_frames_padding_mask = claim_frames_padding_mask
        self.doc_words_mask = doc_words_mask
        self.guid = guid
        self.original_text_b_ids = original_text_b_ids


class InputSeqLabelFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 extraction_mask=None, extraction_start_ids=None, extraction_end_ids=None,
                 augmentation_seq_labels_mask=None, augmentation_seq_label=None, guid=None, original_text_b_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.extraction_mask = extraction_mask
        self.extraction_start_ids = extraction_start_ids
        self.extraction_end_ids = extraction_end_ids
        self.augmentation_seq_labels_mask = augmentation_seq_labels_mask
        self.augmentation_seq_label = augmentation_seq_label
        self.guid = guid
        self.original_text_b_ids = original_text_b_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a jsonl file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
        return lines


class FactCCGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["CORRECT", "INCORRECT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            label = example["label"]
            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        return examples


class FactCCAmrMultiLabelGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #return ["Incorrect", "PredE", "EntE", "NumE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "NumE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim_amr"]
            if text_b.strip() == "":
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("Incorrect")
                if "augmentation" in example.keys():
                    augmentation = example["augmentation"]
                    label.append(augmentation)
                elif "tag" in example.keys():
                    tag = example["tag"]
                    if tag.startswith("neg.verb_antonomy"):
                        label.append("neg.verb_antonomy")
                    elif tag.startswith("neg.adj_antonomy"):
                        label.append("neg.adj_antonomy")
                    elif tag.startswith("neg.entity"):
                        label.append("neg.entity")
                    elif tag.startswith("neg.numerical"):
                        label.append("neg.numerical")
                    elif tag.startswith("neg.date"):
                        label.append("neg.date")
            if "augmentation" in example.keys() and example["noise"] == True:
                label.append("word_noise")
            error_counter.update(label)

            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print()
        print("total no. of samples")
        print(len(lines))
        print()
        return examples


class FactCCAmrAugMultiLabelGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #return ["Incorrect", "PredE", "EntE", "NumE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim_amr"]
            if text_b.strip() == "":
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("Incorrect")
                if "augmentation" in example.keys():
                    augmentation = example["augmentation"]
                    #label.append(augmentation)
                    label += augmentation
            error_counter.update(label)

            extraction_span = example["extraction_span"]
            augmentation_span = []

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print()
        print("total no. of samples")
        print(len(lines))
        print()
        return examples


class FactCCAmrAugMultiLabelSeqLabelGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #return ["Incorrect", "PredE", "EntE", "NumE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim_amr"]
            if text_b.strip() == "":
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("Incorrect")
                if "augmentation" in example.keys():
                    augmentation = example["augmentation"]
                    #label.append(augmentation)
                    label += augmentation
            error_counter.update(label)

            extraction_span = example["extraction_span"]
            augmentation_seq_label = example["augmentation_seq_label"]
            augmentation_seq_label_mask = example["augmentation_seq_label_mask"]

            examples.append(
                InputSeqLabelExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_seq_label=augmentation_seq_label, augmentation_seq_label_mask=augmentation_seq_label_mask))
        print("error_counter:")
        print(error_counter)
        print()
        print("total no. of samples")
        print(len(lines))
        print()
        return examples


class FactCCMultiLabelSeqLabelGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #return ["Incorrect", "PredE", "EntE", "NumE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim_tokenized"]
            if text_b == []:
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("Incorrect")
                if "augmentation" in example.keys() and example["augmentation"]:
                    #augmentation = example["augmentation"]
                    #label += augmentation
                    for augmentation in example["augmentation"]:
                        if augmentation.startswith("neg.verb_antonomy"):
                            label.append("neg.verb_antonomy")
                        elif augmentation.startswith("neg.adj_antonomy"):
                            label.append("neg.adj_antonomy")
                        elif augmentation.startswith("neg.entity"):
                            label.append("neg.entity")
                        elif augmentation.startswith("neg.numerical"):
                            label.append("neg.numerical")
                        elif augmentation.startswith("neg.date"):
                            label.append("neg.date")
                        else:
                            label.append(augmentation)
            error_counter.update(label)

            extraction_span = example["extraction_span"]
            augmentation_seq_label = example["augmentation_seq_label"]
            augmentation_seq_label_mask = example["augmentation_seq_label_mask"]

            examples.append(
                InputSeqLabelExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_seq_label=augmentation_seq_label, augmentation_seq_label_mask=augmentation_seq_label_mask))
        print("error_counter:")
        print(error_counter)
        print()
        print("total no. of samples")
        print(len(lines))
        print()
        return examples


class FactCCBinarySeqLabelGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["CORRECT", "INCORRECT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim_tokenized"]
            if text_b == []:
                continue
            binary_label = example["label"]
            label = binary_label
            extraction_span = example["extraction_span"]
            augmentation_seq_label = example["augmentation_seq_label"]
            augmentation_seq_label_mask = example["augmentation_seq_label_mask"]

            examples.append(
                InputSeqLabelExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_seq_label=augmentation_seq_label, augmentation_seq_label_mask=augmentation_seq_label_mask))
        print("total no. of samples")
        print(len(lines))
        print()
        return examples


class FactCCMultiLabelGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            if not text_b:
                skip_counter += 1
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("Incorrect")
                if "augmentation" in example.keys() and example["augmentation"]:
                    #augmentation = example["augmentation"]
                    #label += augmentation
                    for augmentation in example["augmentation"]:
                        if augmentation.startswith("neg.verb_antonomy"):
                            label.append("neg.verb_antonomy")
                        elif augmentation.startswith("neg.adj_antonomy"):
                            label.append("neg.adj_antonomy")
                        elif augmentation.startswith("neg.entity"):
                            label.append("neg.entity")
                        elif augmentation.startswith("neg.numerical"):
                            label.append("neg.numerical")
                        elif augmentation.startswith("neg.date"):
                            label.append("neg.date")
                        else:
                            label.append(augmentation)
            error_counter.update(label)
            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelSyntheticProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        return ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        #augmentation_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            if not text_b:
                skip_counter += 1
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                if "augmentation" in example.keys() and example["augmentation"]:
                    augmentation = example["augmentation"]
                    if augmentation == "EntitySwap" or augmentation == "LocationSwap" or augmentation == "PronounSwap" \
                                or augmentation == "NumberSwap" or augmentation == "DateSwap":
                        label.append("intrinsic-NP")
                    elif augmentation == "NegateSentences" or augmentation == "VerbSwap":
                        label.append("intrinsic-predicate")
                    elif augmentation in ["DateSwapRandomDoc", "NumberSwapRandomDoc", "EntitySwapRandomDoc"]:
                        label.append("extrinsic-NP")
                    elif augmentation == "VerbSwapRandomDoc":
                        label.append("extrinsic-predicate")
                if "tag" in example.keys() and example["tag"]:
                    tag = example["tag"]
                    if tag.startswith("neg.entity") or tag.startswith("neg.date") or \
                                tag.startswith("neg.numerical") or tag.startswith("neg.adj_int_antonomy"):
                        label.append("intrinsic-NP")
                    elif tag.startswith("neg.verb_int_antonomy"):
                        label.append("intrinsic-predicate")
                    elif tag.startswith("neg.verb_ext_antonomy"):
                        label.append("extrinsic-predicate")
                    elif tag.startswith("neg.adj_ext_antonomy"):
                        label.append("extrinsic-NP")
                    else:
                        print("miss tag")
                        print(tag)
                    #augmentation_counter.update(example["augmentation"])
                error_counter.update(label)

            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        #print("augmentation_counter")
        #print(augmentation_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelSyntheticBackupProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        return ["incorrect", "extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        augmentation_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            if not text_b:
                skip_counter += 1
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("incorrect")
                if "augmentation" in example.keys() and example["augmentation"]:
                    #augmentation = example["augmentation"]
                    #label += augmentation
                    for augmentation in example["augmentation"]:
                        if augmentation == "EntitySwap" or augmentation == "LocationSwap" or augmentation == "PronounSwap" \
                                or augmentation == "NumberSwap" or augmentation == "DateSwap" or \
                                augmentation.startswith("neg.entity") or augmentation.startswith("neg.date") or \
                                augmentation.startswith("neg.numerical"):
                            label.append("intrinsic-NP")
                        elif augmentation.startswith("neg.adj_antonomy"):
                            label.append("extrinsic-NP")
                        elif augmentation == "NegateSentences":
                            label.append("intrinsic-predicate")
                        elif augmentation.startswith("neg.verb_antonomy"):
                            label.append("extrinsic-predicate")
                        else:
                            print("augmentation")
                            print(augmentation)
                            raise ValueError
                    augmentation_counter.update(example["augmentation"])
                error_counter.update(label)

            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print("augmentation_counter")
        print(augmentation_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelAnnotatedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["Incorrect", "PredE", "EntE", "CircE"]
        return ["incorrect", "extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["doc"]
            text_b = example["summ"]
            if not text_b:
                skip_counter += 1
                continue
            #binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            raw_labels = example["error_type"]
            label = []
            if "correct" not in raw_labels:
                label.append("incorrect")
                for l in raw_labels:
                    if l in ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]:
                        label.append(l)
            error_counter.update(label)
            extraction_span = None
            augmentation_span = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelAnnotatedWithEntireSentProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["Incorrect", "PredE", "EntE", "CircE"]
        return ["incorrect", "extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["doc"]
            text_b = example["summ"]
            if not text_b:
                skip_counter += 1
                continue
            #binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            raw_labels = example["error_type"]
            label = []
            if "correct" not in raw_labels:
                label.append("incorrect")
                for l in raw_labels:
                    if l in ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]:
                        label.append(l)
                    elif l == "extrinsic-entire_sent":
                        label.append("extrinsic-NP")
                        label.append("extrinsic-predicate")
                    elif l == "intrinsic-entire_sent":
                        label.append("intrinsic-NP")
                        label.append("intrinsic-predicate")
            error_counter.update(label)
            extraction_span = None
            augmentation_span = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelAnnotatedWithEntireSentFourLabelProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["Incorrect", "PredE", "EntE", "CircE"]
        return ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["doc"]
            text_b = example["summ"]
            if not text_b:
                skip_counter += 1
                continue
            #binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            raw_labels = example["error_type"]
            label = []
            if "correct" not in raw_labels:
                #label.append("incorrect")
                for l in raw_labels:
                    if l in ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]:
                        label.append(l)
                    elif l == "extrinsic-entire_sent":
                        label.append("extrinsic-NP")
                        label.append("extrinsic-predicate")
                    elif l == "intrinsic-entire_sent":
                        label.append("intrinsic-NP")
                        label.append("intrinsic-predicate")
                    elif l == "entire_sent":
                        label.append("extrinsic-NP")
                        label.append("extrinsic-predicate")
                        label.append("intrinsic-NP")
                        label.append("intrinsic-predicate")
            error_counter.update(label)
            extraction_span = None
            augmentation_span = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelClaimAttnSyntheticProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        return ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        augmentation_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            if not text_b or not example["doc_frames_word_ids"]:
                skip_counter += 1
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                if "augmentation" in example.keys() and example["augmentation"]:
                    augmentation = example["augmentation"]
                    if augmentation == "EntitySwap" or augmentation == "LocationSwap" or augmentation == "PronounSwap" \
                            or augmentation == "NumberSwap" or augmentation == "DateSwap":
                        label.append("intrinsic-NP")
                    elif augmentation == "NegateSentences" or augmentation == "VerbSwap":
                        label.append("intrinsic-predicate")
                    elif augmentation in ["DateSwapRandomDoc", "NumberSwapRandomDoc", "EntitySwapRandomDoc"]:
                        label.append("extrinsic-NP")
                    elif augmentation == "VerbSwapRandomDoc":
                        label.append("extrinsic-predicate")
                if "tag" in example.keys() and example["tag"]:
                    tag = example["tag"]
                    if tag.startswith("neg.entity") or tag.startswith("neg.date") or \
                            tag.startswith("neg.numerical") or tag.startswith("neg.adj_int_antonomy"):
                        label.append("intrinsic-NP")
                    elif tag.startswith("neg.verb_int_antonomy"):
                        label.append("intrinsic-predicate")
                    elif tag.startswith("neg.verb_ext_antonomy"):
                        label.append("extrinsic-predicate")
                    elif tag.startswith("neg.adj_ext_antonomy"):
                        label.append("extrinsic-NP")
                    else:
                        print("miss tag")
                        print(tag)
                    # augmentation_counter.update(example["augmentation"])
                error_counter.update(label)

            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputAttnExample(guid=len(examples), text_a=text_a, text_b=text_b, label=label,
                             doc_frames_word_ids=example["doc_frames_word_ids"], claim_frames_word_ids=example["claim_frames_word_ids"]))
        print("error_counter:")
        print(error_counter)
        #print("augmentation_counter")
        #print(augmentation_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelClaimAttnAnnotatedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["Incorrect", "PredE", "EntE", "CircE"]
        return ["incorrect", "extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["doc"]
            text_b = example["summ"]
            if not text_b:
                skip_counter += 1
                continue
            #binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            raw_labels = example["error_type"]
            label = []
            if "correct" not in raw_labels:
                label.append("incorrect")
                for l in raw_labels:
                    if l in ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]:
                        label.append(l)
            error_counter.update(label)
            extraction_span = None
            augmentation_span = None

            examples.append(
                InputAttnExample(guid=len(examples), text_a=text_a, text_b=text_b, label=label,
                                 doc_frames_word_ids=example["doc_frames_word_ids"],
                                 claim_frames_word_ids=example["claim_frames_word_ids"]))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelClaimAttnAnnotatedWithEntireSentProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["Incorrect", "PredE", "EntE", "CircE"]
        return ["incorrect", "extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["doc"]
            text_b = example["summ"]
            if not text_b:
                skip_counter += 1
                continue
            #binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            raw_labels = example["error_type"]
            label = []
            if "correct" not in raw_labels:
                label.append("incorrect")
                for l in raw_labels:
                    if l in ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]:
                        label.append(l)
                    elif l == "extrinsic-entire_sent":
                        label.append("extrinsic-NP")
                        label.append("extrinsic-predicate")
                    elif l == "intrinsic-entire_sent":
                        label.append("intrinsic-NP")
                        label.append("intrinsic-predicate")
            error_counter.update(label)
            extraction_span = None
            augmentation_span = None

            examples.append(
                InputAttnExample(guid=len(examples), text_a=text_a, text_b=text_b, label=label,
                                 doc_frames_word_ids=example["doc_frames_word_ids"],
                                 claim_frames_word_ids=example["claim_frames_word_ids"]))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class AggreFactMultiLabelClaimAttnAnnotatedWithEntireSentFourLabelProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["Incorrect", "PredE", "EntE", "CircE"]
        return ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["doc"]
            text_b = example["summ"]
            if not text_b:
                skip_counter += 1
                continue
            #binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            raw_labels = example["error_type"]
            label = []
            if "correct" not in raw_labels:
                #label.append("incorrect")
                for l in raw_labels:
                    if l in ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]:
                        label.append(l)
                    elif l == "extrinsic-entire_sent":
                        label.append("extrinsic-NP")
                        label.append("extrinsic-predicate")
                    elif l == "intrinsic-entire_sent":
                        label.append("intrinsic-NP")
                        label.append("intrinsic-predicate")
                    """
                    elif l == "entire_sent":
                        label.append("extrinsic-NP")
                        label.append("extrinsic-predicate")
                        label.append("intrinsic-NP")
                        label.append("intrinsic-predicate")
                    """
            error_counter.update(label)
            extraction_span = None
            augmentation_span = None

            examples.append(
                InputAttnExample(guid=len(examples), text_a=text_a, text_b=text_b, label=label,
                                 doc_frames_word_ids=example["doc_frames_word_ids"],
                                 claim_frames_word_ids=example["claim_frames_word_ids"]))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class FactCCMultiLabelAttnGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["Incorrect", "PredE", "EntE", "CircE"]
        #return ["Incorrect", "extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        skip_counter = 0
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            if not text_b:
                skip_counter += 1
                continue
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("Incorrect")
                if "augmentation" in example.keys() and example["augmentation"]:
                    #augmentation = example["augmentation"]
                    #label += augmentation
                    for augmentation in example["augmentation"]:
                        if augmentation.startswith("neg.verb_antonomy"):
                            label.append("neg.verb_antonomy")
                        elif augmentation.startswith("neg.adj_antonomy"):
                            label.append("neg.adj_antonomy")
                        elif augmentation.startswith("neg.entity"):
                            label.append("neg.entity")
                        elif augmentation.startswith("neg.numerical"):
                            label.append("neg.numerical")
                        elif augmentation.startswith("neg.date"):
                            label.append("neg.date")
                        else:
                            label.append(augmentation)
            error_counter.update(label)
            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]
            doc_srl_verb_word_ids = example["doc_srl_verb_word_ids_dict"]
            claim_srl_verb_word_ids = example["claim_srl_verb_word_ids_dict"]
            examples.append(
                InputAttnExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                 doc_srl_verb_word_ids=doc_srl_verb_word_ids, claim_srl_verb_word_ids=claim_srl_verb_word_ids))
        print("error_counter:")
        print(error_counter)
        print("total no. of samples")
        print(len(lines))
        print("skipped:")
        print(skip_counter)
        print()
        return examples


class FactCCMultiLabelGeneratedProcessorBackup(DataProcessor):
    """Processor for the generated FactCC data set."""
    """
    def __init__(self):
        super().__init__()
        # [predicate error, entity error, coreference error, grammatical error]
        self.error_type_to_label_id = {"noise": 3}
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        error_counter = Counter()
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            binary_label = example["label"]
            #label = [0] * 4  # [predicate error, entity error, coreference error, grammatical error]
            label = []
            if binary_label == "INCORRECT":
                label.append("Incorrect")
                if "augmentation" in example.keys():
                    augmentation = example["augmentation"]
                    """
                    if augmentation == "NegateSentences":
                        label.append("PredE")
                    elif augmentation == "EntitySwap":
                        label.append("EntE")
                    elif augmentation == "NumberSwap":
                        label.append("NumE")
                    elif augmentation == "PronounSwap":
                        label.append("CorefE")
                    elif augmentation == "DateSwap":
                        label.append("DateE")
                    """
                    label.append(augmentation)
                elif "tag" in example.keys():
                    tag = example["tag"]
                    """
                    if tag.startswith("neg.verb_antonomy") or tag.startswith("neg.adj_antonomy"):
                        label.append("PredE")
                    elif tag.startswith("neg.entity"):
                        label.append("EntE")
                    elif tag.startswith("neg.numerical"):
                        label.append("NumE")
                    """
                    if tag.startswith("neg.verb_antonomy"):
                        label.append("neg.verb_antonomy")
                    elif tag.startswith("neg.adj_antonomy"):
                        label.append("neg.adj_antonomy")
                    elif tag.startswith("neg.entity"):
                        label.append("neg.entity")
                    elif tag.startswith("neg.numerical"):
                        label.append("neg.numerical")
                    elif tag.startswith("neg.date"):
                        label.append("neg.date")
            if "augmentation" in example.keys() and example["noise"] == True:
                label.append("word_noise")
            error_counter.update(label)
            """
            print(label)
            if "augmentation" in example.keys():
                print("augmentation")
                print(example["augmentation"])
            if "tag" in example.keys():
                print("tag")
                print(example["tag"])
            print()
            """

            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        print("error_counter:")
        print(error_counter)
        print()
        print("total no. of samples")
        print(len(lines))
        print()
        return examples


class FrankMultiLabelManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            text_a = example["article"]
            for summary_sent, sent_annotations in zip(example["summary_sentences"], example['summary_sentences_annotations']):
                text_b = summary_sent
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, original_text_b=summary_sent))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FrankMultiLabelFilteredManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            text_a = example["article"]
            for summary_sent, sent_annotations, article_filtered in zip(example["summary_sentences"],
                    example['summary_sentences_annotations'], example["article"]):
                text_a = article_filtered
                text_b = summary_sent
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, original_text_b=summary_sent))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FrankMultiLabelSRLManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence_srl_filtered.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            #text_a = example["article"]
            for summary_sent_srl, summary_sent, sent_annotations, article_filtered in zip(example["summary_sentences_srl"], example["summary_sentences"], example['summary_sentences_annotations'], example["article"]):
                text_a = article_filtered
                text_b = summary_sent_srl
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, original_text_b=summary_sent))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FrankMultiLabelSRLNestedManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence_srl_nested_filtered.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            for summary_sent_srl, summary_sent, sent_annotations, article_filtered in zip(example["summary_sentences_srl"], example["summary_sentences"], example['summary_sentences_annotations'], example["article"]):
                text_a = article_filtered
                text_b = summary_sent_srl
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, original_text_b=summary_sent))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FrankMultiLabelAttnManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence-claim-doc-srl-verb-ids.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            text_a = example["article"]
            doc_srl_verb_word_ids = example["doc_srl_verb_word_ids_dict"]
            for summary_sent, sent_annotations, claim_srl_verb_word_ids in zip(example["summary_sentences"], example['summary_sentences_annotations'], example['claim_srl_verb_word_ids_dict']):
                text_b = summary_sent
                #print("create sample text_b")
                #print(text_b)
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(
                    InputAttnExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                     doc_srl_verb_word_ids=doc_srl_verb_word_ids,
                                     claim_srl_verb_word_ids=claim_srl_verb_word_ids))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FrankAmrMultiLabelManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence_amr.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            text_a = example["article"]
            for summary_sent_amr, summary_sent, sent_annotations in zip(example["summary_sentences_amr"], example["summary_sentences"], example['summary_sentences_annotations']):
                text_b = summary_sent_amr
                #print("create sample text_b")
                #print(text_b)
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, original_text_b=summary_sent))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FrankAmrMultiLabelSeqLabelManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence_amr.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            text_a = example["article"]
            for summary_sent_amr, summary_sent, sent_annotations in zip(example["summary_sentences_amr"], example["summary_sentences"], example['summary_sentences_annotations']):
                text_b = summary_sent_amr
                #print("create sample text_b")
                #print(text_b)
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(InputSeqLabelExample(guid=guid, text_a=text_a, text_b=text_b, label=label, original_text_b=summary_sent))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FrankSRLnestedMultiLabelSeqLabelManualProcessor(DataProcessor):
    """Processor for the multilabel frank data set."""
    def get_train_examples(self, data_dir):
        raise ValueError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            json.load(open(os.path.join(data_dir, "human_annotations_sentence_srl_nested_filtered_seq_label.json"))), "dev")

    def get_labels(self):
        """See base class."""
        #return ["PredE", "EntE", "NumE", "CorefE", "GramE"]
        #Counter({'NoE': 1218, 'OutE': 467, 'EntE': 434, 'CircE': 153, 'GramE': 142, 'RelE': 132, 'CorefE': 108, 'LinkE': 39, 'OtherE': 10})
        #return ["Incorrect", "PredE", "EntE", "CircE", "CorefE"]
        return ["Incorrect", "PredE", "EntE", "CircE"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        all_sent_error_counter = Counter()
        examples = []
        num_summaries = 0
        for example in lines:
            #guid = example["hash"]
            guid = num_summaries
            for summary_sent_srl, summary_sent, sent_annotations, article_filtered, seq_label_mask in zip(example["summary_sentences_srl"], example["summary_sentences"], example['summary_sentences_annotations'], example["article"], example["summary_sentences_seq_label_mask"]):
                text_a = article_filtered
                text_b = summary_sent_srl.split(" ")
                #label = []
                sent_error_counter = Counter()
                # sent_annotations: dict
                for annotator, error_types in sent_annotations.items():
                    sent_error_counter.update(error_types)
                # errors annotated by at least two annotators
                sent_error_list = [error_type for error_type, cnt in sent_error_counter.items() if cnt > 1]
                if "NoE" not in sent_error_list:
                    label = ["Incorrect"] + sent_error_list
                else:
                    label = []
                examples.append(InputSeqLabelExample(guid=guid, text_a=text_a, text_b=text_b, label=label, original_text_b=summary_sent, augmentation_seq_label_mask=seq_label_mask))
                all_sent_error_counter.update(sent_error_list)
            num_summaries += 1
        print("sent total")
        print(sum(all_sent_error_counter.values()))
        print("all_sent_error_counter")
        print(all_sent_error_counter)
        return examples


class FactCCManualProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["CORRECT", "INCORRECT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, example) in enumerate(lines):
            guid = str(i)
            text_a = example["text"]
            text_b = example["claim"]
            label = example["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True
                                 ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    tokens_b_truncated_count = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            #logger.info("token b before trunc")
            #logger.info(" ".join([str(x) for x in tokens_b]))
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            is_tokens_b_truncated = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            if is_tokens_b_truncated:
                tokens_b_truncated_count += 1
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            """
            if sep_token_extra:
                segment_ids += [sequence_a_segment_id] * (len(tokens_b) + 1)
            else:
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            """
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            #logger.info("token b after trunc")
            #logger.info(" ".join([str(x) for x in tokens_b]))

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        ####### AUX LOSS DATA
        # get tokens_a mask
        extraction_span_len = len(tokens_a) + 2
        extraction_mask = [1 if 0 < ix < extraction_span_len else 0 for ix in range(max_seq_length)]

        # get extraction labels
        if example.extraction_span:
            ext_start, ext_end = example.extraction_span
            extraction_start_ids = ext_start + 1
            extraction_end_ids = ext_end + 1
        else:
            extraction_start_ids = extraction_span_len
            extraction_end_ids = extraction_span_len

        augmentation_mask = [1 if extraction_span_len <= ix < extraction_span_len + len(tokens_b) + 1  else 0 for ix in range(max_seq_length)]

        if example.augmentation_span:
            if isinstance(example.augmentation_span[0], list):
                last_sep_token = extraction_span_len + len(tokens_b)
                augmentation_start_ids = last_sep_token
                augmentation_end_ids = last_sep_token
            else:
                aug_start, aug_end = example.augmentation_span
                augmentation_start_ids = extraction_span_len + aug_start
                augmentation_end_ids = extraction_span_len + aug_end
        else:
            last_sep_token = extraction_span_len + len(tokens_b)
            augmentation_start_ids = last_sep_token
            augmentation_end_ids = last_sep_token

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        #print("output_mode")
        #print(output_mode)

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "multi_label_classification":
            label_id = [0] * len(label_map)
            for l in example.label:
                label_id[label_map[l]] = 1
            #print("example.label")
            #print(example.label)
            #print("label_id")
            #print(label_id)
            #print("label_map")
            #print(label_map)
            #print()

            #label_id = multi_label_map(example.label)
            #label_id = multi_label_map_frame_only(example.label)
            #logger.info("example.label")
            #logger.info(example.label)
            #logger.info(label_id)
            #logger.info("")
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("ext mask: %s" % " ".join([str(x) for x in extraction_mask]))
            logger.info("ext start: %d" % extraction_start_ids)
            logger.info("ext end: %d" % extraction_end_ids)
            logger.info("aug mask: %s" % " ".join([str(x) for x in augmentation_mask]))
            logger.info("aug start: %d" % augmentation_start_ids)
            logger.info("aug end: %d" % augmentation_end_ids)
            if output_mode == "multi_label_classification":
                logger.info("label: %s" % " ".join([str(x) for x in label_id]))
            else:
                logger.info("label: %d" % label_id)

        extraction_start_ids = min(extraction_start_ids, 511)
        extraction_end_ids = min(extraction_end_ids, 511)
        augmentation_start_ids = min(augmentation_start_ids, 511)
        augmentation_end_ids = min(augmentation_end_ids, 511)

        # original tokens b
        if example.original_text_b:
            original_text_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.original_text_b))
            original_text_b_ids = original_text_b_ids[:100]
            if len(original_text_b_ids) < 100:
                original_text_b_ids = original_text_b_ids + [pad_token] * (100-len(original_text_b_ids))
        else:
            original_text_b_ids = None

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          extraction_mask=extraction_mask,
                          extraction_start_ids=extraction_start_ids,
                          extraction_end_ids=extraction_end_ids,
                          augmentation_mask=augmentation_mask,
                          augmentation_start_ids=augmentation_start_ids,
                          augmentation_end_ids=augmentation_end_ids,
                          guid=example.guid,
                          original_text_b_ids=original_text_b_ids))

    logger.info("tokens b truncated count: %d" % tokens_b_truncated_count)
    return features


def convert_attn_examples_to_features(example, label_list, max_seq_length,
                                 tokenizer, output_mode, pad_token=0):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    tokenized_input_text = tokenizer(text=example.text_a.split(" "), text_pair=example.text_b.split(" "), padding="max_length",
                                     is_split_into_words=True, truncation="longest_first", max_length=max_seq_length)
    input_word_ids = tokenized_input_text.word_ids()
    input_ids = tokenized_input_text.input_ids
    input_mask = tokenized_input_text.attention_mask
    segment_ids = tokenized_input_text.token_type_ids

    # construct mask
    doc_srl_verb_word_ids = example.doc_srl_verb_word_ids
    claim_srl_verb_word_ids = example.claim_srl_verb_word_ids
    # convert key from str to int
    doc_srl_verb_word_ids_old_keys = list(doc_srl_verb_word_ids.keys())
    claim_srl_verb_word_ids_old_keys = list(claim_srl_verb_word_ids.keys())
    for k in doc_srl_verb_word_ids_old_keys:
        doc_srl_verb_word_ids[int(k)] = doc_srl_verb_word_ids[k]
        del doc_srl_verb_word_ids[k]
    for k in claim_srl_verb_word_ids_old_keys:
        claim_srl_verb_word_ids[int(k)] = claim_srl_verb_word_ids[k]
        del claim_srl_verb_word_ids[k]
    #rint("doc_srl_verb_word_ids_old_keys")
    #print(doc_srl_verb_word_ids_old_keys)

    #verb_bpe_ids = []
    cls_attn_mask = []
    verb_attn_mask_2d = []
    for i, (word_id, segment_id) in enumerate(zip(input_word_ids, segment_ids)):
        if i == 0 or (word_id in doc_srl_verb_word_ids and segment_id==0) or (word_id in claim_srl_verb_word_ids and segment_id==1):
        #if (word_id in doc_srl_verb_word_ids and segment_id==0) or (word_id in claim_srl_verb_word_ids and segment_id==1):
            cls_attn_mask.append(0)  # 0 means allow attention in multihead_attention class of pytorch
        else:
            cls_attn_mask.append(1)
        #print("word_id")
        #print(word_id)
        #print("segment_id")
        #print(segment_id)

        if (word_id in doc_srl_verb_word_ids and segment_id==0) or (word_id in claim_srl_verb_word_ids and segment_id==1):
            if segment_id == 0:
                arg_word_ids = doc_srl_verb_word_ids[word_id]["ARG"]
                argm_word_ids = doc_srl_verb_word_ids[word_id]["ARGM"]
            else:
                arg_word_ids = claim_srl_verb_word_ids[word_id]["ARG"]
                argm_word_ids = claim_srl_verb_word_ids[word_id]["ARGM"]
            verb_attn_mask_1d = []
            for word_id_j, segment_id_j in zip(input_word_ids, segment_ids):
                if segment_id_j == segment_id and (word_id_j in arg_word_ids or word_id_j in argm_word_ids or word_id_j == word_id):
                    verb_attn_mask_1d.append(0)  # 0 means allow attention in multihead_attention class of pytorch
                else:
                    verb_attn_mask_1d.append(1)
            verb_attn_mask_2d.append(verb_attn_mask_1d)
        else:
            verb_attn_mask_1d = [1] * len(input_word_ids)
            verb_attn_mask_1d[i] = 0
            verb_attn_mask_2d.append(verb_attn_mask_1d)
    #print("output_mode")
    #print(output_mode)
    #augmentation_mask = [1 if segment_ids[ix] == 1 and input_word_ids[ix] is not None else 0 for ix in
    #                     range(len(input_word_ids))]
    #extraction_mask = [1 if segment_ids[ix] == 0 and input_word_ids[ix] is not None else 0 for ix in
    #                     range(len(input_word_ids))]

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "multi_label_classification":
        #label_id = multi_label_map(example.label)
        label_id = multi_label_map_frame_only(example.label)
        #logger.info("example.label")
        #logger.info(example.label)
        #label_id = multi_label_map_frame_only_amr(example.label)
        #logger.info(label_id)
        #logger.info("")
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    # original tokens b
    if example.original_text_b:
        original_text_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.original_text_b))
        original_text_b_ids = original_text_b_ids[:100]
        if len(original_text_b_ids) < 100:
            original_text_b_ids = original_text_b_ids + [pad_token] * (100-len(original_text_b_ids))
    else:
        original_text_b_ids = None

    feature = InputAttnFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label_id=label_id,
                      verb_attn_mask=verb_attn_mask_2d,
                      cls_attn_mask=cls_attn_mask,
                      guid=example.guid,
                      original_text_b_ids=original_text_b_ids)

    return feature


def convert_claim_attn_examples_to_features(example, label_list, max_seq_length,
                                 tokenizer, output_mode, pad_token=0):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    tokenized_input_text = tokenizer(text=example.text_a.split(" "), text_pair=example.text_b.split(" "), padding="max_length",
                                     is_split_into_words=True, truncation="longest_first", max_length=max_seq_length)
    input_word_ids = tokenized_input_text.word_ids()
    input_ids = tokenized_input_text.input_ids
    input_mask = tokenized_input_text.attention_mask
    segment_ids = tokenized_input_text.token_type_ids

    # construct mask
    doc_frames_word_ids = example.doc_frames_word_ids
    claim_frames_word_ids = example.claim_frames_word_ids

    doc_frames_word_mask_2d = []  # [n_doc_frames, seq_len]
    doc_frames_np_word_mask_2d = []  # [n_doc_frames, seq_len]
    doc_frames_predicate_word_mask_2d = []  # [n_doc_frames, seq_len]
    claim_frames_word_mask_2d = []  # [n_doc_frames, seq_len]
    claim_frames_np_word_mask_2d = []  # [n_doc_frames, seq_len]
    claim_frames_predicate_word_mask_2d = []  # [n_doc_frames, seq_len]

    for doc_frame in doc_frames_word_ids:
        doc_frames_word_mask_1d = []
        doc_frames_np_word_mask_1d = []
        doc_frames_predicate_word_mask_1d = []
        for i, (word_id, segment_id) in enumerate(zip(input_word_ids, segment_ids)):
            if segment_id == 0 and (word_id in doc_frame["V"] or word_id in doc_frame["NP"]):
                doc_frames_word_mask_1d.append(1)  # 1 indicates positive
                if word_id in doc_frame["V"]:
                    doc_frames_predicate_word_mask_1d.append(1)
                else:
                    doc_frames_predicate_word_mask_1d.append(0)
                if word_id in doc_frame["NP"]:
                    doc_frames_np_word_mask_1d.append(1)
                else:
                    doc_frames_np_word_mask_1d.append(0)
            else:
                doc_frames_word_mask_1d.append(0)  # 0 indicates negative
                doc_frames_predicate_word_mask_1d.append(0)
                doc_frames_np_word_mask_1d.append(0)
        if any(doc_frames_word_mask_1d):
            doc_frames_word_mask_2d.append(doc_frames_word_mask_1d)
            doc_frames_np_word_mask_2d.append(doc_frames_np_word_mask_1d)
            doc_frames_predicate_word_mask_2d.append(doc_frames_predicate_word_mask_1d)
        #print("doc_frames_word_mask_1d")
        #print(doc_frames_word_mask_1d)
    #print()

    # ensure doc_frames_word_ids is non empty
    assert doc_frames_word_ids

    for claim_frame in claim_frames_word_ids:
        claim_frames_word_mask_1d = []
        claim_frames_np_word_mask_1d = []
        claim_frames_predicate_word_mask_1d = []
        for i, (word_id, segment_id) in enumerate(zip(input_word_ids, segment_ids)):
            if segment_id == 1 and (word_id in claim_frame["V"] or word_id in claim_frame["NP"]):
                claim_frames_word_mask_1d.append(1)  # 1 indicates positive
                if word_id in claim_frame["V"]:
                    claim_frames_predicate_word_mask_1d.append(1)
                else:
                    claim_frames_predicate_word_mask_1d.append(0)
                if word_id in claim_frame["NP"]:
                    claim_frames_np_word_mask_1d.append(1)
                else:
                    claim_frames_np_word_mask_1d.append(0)
            else:
                claim_frames_word_mask_1d.append(0)  # 0 indicates negative
                claim_frames_np_word_mask_1d.append(0)  # 0 indicates negative
                claim_frames_predicate_word_mask_1d.append(0)  # 0 indicates negative
        if any(claim_frames_word_mask_1d):
            claim_frames_word_mask_2d.append(claim_frames_word_mask_1d)
            claim_frames_np_word_mask_2d.append(claim_frames_word_mask_1d)
            claim_frames_predicate_word_mask_2d.append(claim_frames_word_mask_1d)
        #print("claim_frames_word_mask_2d")
        #print(claim_frames_word_mask_2d)
    #print()

    # if cannot detect any claim frames, use the mean pooling of the entire claim
    if not claim_frames_word_ids:
        claim_frames_word_mask_1d = []
        claim_frames_np_word_mask_1d = []
        claim_frames_predicate_word_mask_1d = []
        for word_id, segment_id in zip(input_word_ids, segment_ids):
            if segment_id == 1 and word_id is not None:
                claim_frames_word_mask_1d.append(1)
                claim_frames_np_word_mask_1d.append(1)
                claim_frames_predicate_word_mask_1d.append(1)
            else:
                claim_frames_word_mask_1d.append(0)
                claim_frames_np_word_mask_1d.append(0)
                claim_frames_predicate_word_mask_1d.append(0)
        assert any(claim_frames_word_mask_1d)
        claim_frames_word_mask_2d.append(claim_frames_word_mask_1d)
        claim_frames_np_word_mask_2d.append(claim_frames_np_word_mask_1d)
        claim_frames_predicate_word_mask_2d.append(claim_frames_predicate_word_mask_1d)

    # padding to doc frames and claim frames
    max_doc_frames_len = int(max_seq_length / 4)
    max_claim_frames_len = int(max_seq_length / 8)
    assert len(doc_frames_word_mask_2d) <= max_doc_frames_len
    assert len(doc_frames_np_word_mask_2d) <= max_doc_frames_len
    assert len(doc_frames_predicate_word_mask_2d) <= max_doc_frames_len
    assert len(claim_frames_word_mask_2d) <= max_claim_frames_len
    assert len(claim_frames_np_word_mask_2d) <= max_claim_frames_len
    assert len(claim_frames_predicate_word_mask_2d) <= max_claim_frames_len

    # [batch, n_doc_frames]
    doc_frames_padding_size = max_doc_frames_len - len(doc_frames_word_mask_2d)
    claim_frames_padding_size = max_claim_frames_len - len(claim_frames_word_mask_2d)
    claim_attn_mask = [0] * len(doc_frames_word_mask_2d) + [1] * doc_frames_padding_size

    claim_frames_padding_mask = [1] * len(claim_frames_word_mask_2d) + [0] * claim_frames_padding_size

    #print("claim_attn_mask")
    #print(claim_attn_mask)
    #print("claim_frames_padding_mask")
    #print(claim_frames_padding_mask)

    # construct doc_frames_word_mask
    if doc_frames_padding_size > 0:
        for i in range(doc_frames_padding_size):
            doc_frames_word_mask_2d.append([0] * max_seq_length)
            doc_frames_np_word_mask_2d.append([0] * max_seq_length)
            doc_frames_predicate_word_mask_2d.append([0] * max_seq_length)

    if claim_frames_padding_size > 0:
        for i in range(claim_frames_padding_size):
            claim_frames_word_mask_2d.append([0] * max_seq_length)
            claim_frames_np_word_mask_2d.append([0] * max_seq_length)
            claim_frames_predicate_word_mask_2d.append([0] * max_seq_length)

    # mask for doc words
    doc_words_mask = []
    for i, (word_id, segment_id) in enumerate(zip(input_word_ids, segment_ids)):
        if segment_id == 0 and word_id is not None:
            doc_words_mask.append(1)
        else:
            doc_words_mask.append(0)

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "multi_label_classification":
        label_id = [0] * len(label_map)
        for l in example.label:
            label_id[label_map[l]] = 1
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    # original tokens b
    if example.original_text_b:
        original_text_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.original_text_b))
        original_text_b_ids = original_text_b_ids[:100]
        if len(original_text_b_ids) < 100:
            original_text_b_ids = original_text_b_ids + [pad_token] * (100-len(original_text_b_ids))
    else:
        original_text_b_ids = None

    feature = InputAttnFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label_id=label_id,
                      doc_frames_word_mask=doc_frames_word_mask_2d,
                      doc_frames_np_word_mask=doc_frames_np_word_mask_2d,
                      doc_frames_predicate_word_mask=doc_frames_predicate_word_mask_2d,
                      claim_frames_word_mask=claim_frames_word_mask_2d,
                      claim_frames_np_word_mask=claim_frames_np_word_mask_2d,
                      claim_frames_predicate_word_mask=claim_frames_predicate_word_mask_2d,
                      claim_attn_mask=claim_attn_mask,
                      claim_frames_padding_mask=claim_frames_padding_mask,
                      doc_words_mask=doc_words_mask,
                      guid=example.guid,
                      original_text_b_ids=original_text_b_ids)

    return feature


def convert_claim_attn_examples_to_features_backup(example, label_list, max_seq_length,
                                 tokenizer, output_mode, pad_token=0):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    tokenized_input_text = tokenizer(text=example.text_a.split(" "), text_pair=example.text_b.split(" "), padding="max_length",
                                     is_split_into_words=True, truncation="longest_first", max_length=max_seq_length)
    input_word_ids = tokenized_input_text.word_ids()
    input_ids = tokenized_input_text.input_ids
    input_mask = tokenized_input_text.attention_mask
    segment_ids = tokenized_input_text.token_type_ids

    # construct mask
    doc_frames_word_ids = example.doc_frames_word_ids
    claim_frames_word_ids = example.claim_frames_word_ids

    doc_frames_word_mask_2d = []  # [n_doc_frames, seq_len]
    claim_frames_word_mask_2d = [[1] + [0] * 511]  # reserve a claim semantic frame for the CLS token, [n_claim_frames, seq_len]

    for doc_frame in doc_frames_word_ids:
        doc_frames_word_mask_1d = []
        for i, (word_id, segment_id) in enumerate(zip(input_word_ids, segment_ids)):
            if segment_id == 0 and (word_id in doc_frame["V"] or word_id in doc_frame["NP"]):
                doc_frames_word_mask_1d.append(1)  # 1 indicates positive
            else:
                doc_frames_word_mask_1d.append(0)  # 0 indicates negative
        if any(doc_frames_word_mask_1d):
            doc_frames_word_mask_2d.append(doc_frames_word_mask_1d)
        #print("doc_frames_word_mask_1d")
        #print(doc_frames_word_mask_1d)
    #print()

    for claim_frame in claim_frames_word_ids:
        claim_frames_word_mask_1d = []
        for i, (word_id, segment_id) in enumerate(zip(input_word_ids, segment_ids)):
            if segment_id == 1 and (word_id in claim_frame["V"] or word_id in claim_frame["NP"]):
                claim_frames_word_mask_1d.append(1)  # 1 indicates positive
            else:
                claim_frames_word_mask_1d.append(0)  # 0 indicates negative
        if any(claim_frames_word_mask_1d):
            claim_frames_word_mask_2d.append(claim_frames_word_mask_1d)
        #print("claim_frames_word_mask_2d")
        #print(claim_frames_word_mask_2d)
    #print()

    # padding to doc frames and claim frames
    max_doc_frames_len = int(max_seq_length / 4)
    max_claim_frames_len = int(max_seq_length / 8)
    assert len(doc_frames_word_mask_2d) <= max_doc_frames_len
    assert len(claim_frames_word_mask_2d) <= max_claim_frames_len

    # [batch, n_doc_frames]
    doc_frames_padding_size = max_doc_frames_len - len(doc_frames_word_mask_2d)
    claim_frames_padding_size = max_claim_frames_len - len(claim_frames_word_mask_2d)
    claim_attn_mask = [0] * len(doc_frames_word_mask_2d) + [1] * doc_frames_padding_size
    """
    claim_attn_mask = []
    for i in range(max_claim_frames_len):
        if i < len(claim_frames_word_mask_2d):
            claim_attn_mask.append([0] * len(doc_frames_word_mask_2d) + [1] * doc_frames_padding_size)
        else:
            claim_attn_mask.append([1] * max_doc_frames_len)
    """
    claim_frames_padding_mask = [1] * len(claim_frames_word_mask_2d) + [0] * claim_frames_padding_size

    #print("claim_attn_mask")
    #print(claim_attn_mask)
    #print("claim_frames_padding_mask")
    #print(claim_frames_padding_mask)

    # construct doc_frames_word_mask
    if doc_frames_padding_size > 0:
        for i in range(doc_frames_padding_size):
            doc_frames_word_mask_2d.append([0] * max_seq_length)

    if claim_frames_padding_size > 0:
        for i in range(claim_frames_padding_size):
            claim_frames_word_mask_2d.append([0] * max_seq_length)

    #print("len(doc_frames_word_mask_2d)")
    #print(len(doc_frames_word_mask_2d))
    #print("len(claim_frames_word_mask_2d)")
    #print(len(claim_frames_word_mask_2d))
    #exit()

    #print("output_mode")
    #print(output_mode)
    #augmentation_mask = [1 if segment_ids[ix] == 1 and input_word_ids[ix] is not None else 0 for ix in
    #                     range(len(input_word_ids))]
    #extraction_mask = [1 if segment_ids[ix] == 0 and input_word_ids[ix] is not None else 0 for ix in
    #                     range(len(input_word_ids))]

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "multi_label_classification":
        label_id = [0] * len(label_map)
        for l in example.label:
            label_id[label_map[l]] = 1
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    # original tokens b
    if example.original_text_b:
        original_text_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.original_text_b))
        original_text_b_ids = original_text_b_ids[:100]
        if len(original_text_b_ids) < 100:
            original_text_b_ids = original_text_b_ids + [pad_token] * (100-len(original_text_b_ids))
    else:
        original_text_b_ids = None

    feature = InputAttnFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label_id=label_id,
                      doc_frames_word_mask=doc_frames_word_mask_2d,
                      claim_frames_word_mask=claim_frames_word_mask_2d,
                      claim_attn_mask=claim_attn_mask,
                      claim_frames_padding_mask=claim_frames_padding_mask,
                      guid=example.guid,
                      original_text_b_ids=original_text_b_ids)

    return feature


def convert_attn_examples_to_features_backup(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True
                                 ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    tokens_b_truncated_count = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #print("text_a")
        #print(example.text_a)
        #print("text_b")
        #print(example.text_b)
        tokenized_input_text = tokenizer(text=example.text_a.split(" "), text_pair=example.text_b.split(" "), padding="max_length",
                                         is_split_into_words=True, truncation="longest_first")
        input_word_ids = tokenized_input_text.word_ids()
        input_ids = tokenized_input_text.input_ids
        input_mask = tokenized_input_text.attention_mask
        segment_ids = tokenized_input_text.token_type_ids

        # construct mask
        doc_srl_verb_word_ids = example.doc_srl_verb_word_ids
        claim_srl_verb_word_ids = example.claim_srl_verb_word_ids
        # fix int key issue
        doc_srl_verb_word_ids_old_keys = list(doc_srl_verb_word_ids.keys())
        claim_srl_verb_word_ids_old_keys = list(claim_srl_verb_word_ids.keys())
        for k in doc_srl_verb_word_ids_old_keys:
            doc_srl_verb_word_ids[int(k)] = doc_srl_verb_word_ids[k]
            del doc_srl_verb_word_ids[k]
        for k in claim_srl_verb_word_ids_old_keys:
            claim_srl_verb_word_ids[int(k)] = claim_srl_verb_word_ids[k]
            del claim_srl_verb_word_ids[k]
        #rint("doc_srl_verb_word_ids_old_keys")
        #print(doc_srl_verb_word_ids_old_keys)

        #verb_bpe_ids = []
        cls_attn_mask = []
        verb_attn_mask_2d = []
        for i, (word_id, segment_id) in enumerate(zip(input_word_ids, segment_ids)):
            if i == 0 or (word_id in doc_srl_verb_word_ids and segment_id==0) or (word_id in claim_srl_verb_word_ids and segment_id==1):
            #if (word_id in doc_srl_verb_word_ids and segment_id==0) or (word_id in claim_srl_verb_word_ids and segment_id==1):
                cls_attn_mask.append(0)  # 0 means allow attention in multihead_attention class of pytorch
            else:
                cls_attn_mask.append(1)
            #print("word_id")
            #print(word_id)
            #print("segment_id")
            #print(segment_id)

            if (word_id in doc_srl_verb_word_ids and segment_id==0) or (word_id in claim_srl_verb_word_ids and segment_id==1):
                if segment_id == 0:
                    arg_word_ids = doc_srl_verb_word_ids[word_id]["ARG"]
                    argm_word_ids = doc_srl_verb_word_ids[word_id]["ARGM"]
                else:
                    arg_word_ids = claim_srl_verb_word_ids[word_id]["ARG"]
                    argm_word_ids = claim_srl_verb_word_ids[word_id]["ARGM"]
                verb_attn_mask_1d = []
                for word_id_j, segment_id_j in zip(input_word_ids, segment_ids):
                    if segment_id_j == segment_id and (word_id_j in arg_word_ids or word_id_j in argm_word_ids or word_id_j == word_id):
                        verb_attn_mask_1d.append(0)  # 0 means allow attention in multihead_attention class of pytorch
                    else:
                        verb_attn_mask_1d.append(1)
                verb_attn_mask_2d.append(verb_attn_mask_1d)
            else:
                verb_attn_mask_1d = [1] * len(input_word_ids)
                verb_attn_mask_1d[i] = 0
                verb_attn_mask_2d.append(verb_attn_mask_1d)
        #print("output_mode")
        #print(output_mode)
        #augmentation_mask = [1 if segment_ids[ix] == 1 and input_word_ids[ix] is not None else 0 for ix in
        #                     range(len(input_word_ids))]
        #extraction_mask = [1 if segment_ids[ix] == 0 and input_word_ids[ix] is not None else 0 for ix in
        #                     range(len(input_word_ids))]

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "multi_label_classification":
            #label_id = multi_label_map(example.label)
            label_id = multi_label_map_frame_only(example.label)
            #logger.info("example.label")
            #logger.info(example.label)
            #label_id = multi_label_map_frame_only_amr(example.label)
            #logger.info(label_id)
            #logger.info("")
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            #logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("cls_attn_mask: %s" % " ".join([str(x) for x in cls_attn_mask]))
            logger.info("aug verb_attn_mask: %s" % " ".join([str(x) for x in verb_attn_mask_2d[0]]))
            if output_mode == "multi_label_classification":
                logger.info("label: %s" % " ".join([str(x) for x in label_id]))
            else:
                logger.info("label: %d" % label_id)

        #extraction_start_ids = min(extraction_start_ids, 511)
        #extraction_end_ids = min(extraction_end_ids, 511)
        #augmentation_start_ids = min(augmentation_start_ids, 511)
        #augmentation_end_ids = min(augmentation_end_ids, 511)

        # original tokens b
        if example.original_text_b:
            original_text_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.original_text_b))
            original_text_b_ids = original_text_b_ids[:100]
            if len(original_text_b_ids) < 100:
                original_text_b_ids = original_text_b_ids + [pad_token] * (100-len(original_text_b_ids))
        else:
            original_text_b_ids = None

        features.append(
            InputAttnFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          verb_attn_mask=verb_attn_mask_2d,
                          cls_attn_mask=cls_attn_mask,
                          guid=example.guid,
                          original_text_b_ids=original_text_b_ids))

    logger.info("tokens b truncated count: %d" % tokens_b_truncated_count)
    return features


def convert_seq_label_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True
                                 ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        #print("tokens_a")
        #print(tokens_a)

        tokens_b = None
        if example.text_b:
            #word_tokenized_text_b = example.text_b.split(" ")
            word_tokenized_text_b = example.text_b
            tokenized_text_b = tokenizer(word_tokenized_text_b, add_special_tokens=False, padding=False, is_split_into_words=True, return_attention_mask=False)
            tokens_b = tokenized_text_b.tokens()
            #print("tokens_b")
            #print(tokens_b)
            #tokens_b = tokenized_text_b["input_ids"]
            #logger.info("token b before trunc")
            #logger.info(" ".join([str(x) for x in tokens_b]))
            # construct seq label mask
            augmentation_seq_label_mask_word_level = example.augmentation_seq_label_mask
            #augmentation_seq_label_mask_word_level = [1] * len(word_tokenized_text_b)
            """
            augmentation_seq_label_mask_word_level = []
            for word in word_tokenized_text_b:
                if word in [")", "("] or word.startswith(":"):
                    augmentation_seq_label_mask_word_level.append(0)
                else:
                    augmentation_seq_label_mask_word_level.append(1)
            """
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            is_tokens_b_truncated = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        #tokenized_text_word_ids = [-100] * len(tokens)
        augmentation_seq_labels_mask = [0] * len(tokens)
        augmentation_seq_labels_bpe = [-100] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            if sep_token_extra:
                segment_ids += [sequence_a_segment_id] * (len(tokens_b) + 1)
            else:
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            #logger.info("token b after trunc")
            #logger.info(" ".join([str(x) for x in tokens_b]))

        tokenized_text_b_word_ids = tokenized_text_b.word_ids()[:len(tokens_b)]
        previous_word_idx = None

        for word_idx in tokenized_text_b_word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                augmentation_seq_labels_mask.append(0)
                augmentation_seq_labels_bpe.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if augmentation_seq_label_mask_word_level:
                    augmentation_seq_labels_mask.append(augmentation_seq_label_mask_word_level[word_idx])
                else:
                    augmentation_seq_labels_mask.append(1)
                """
                except:
                    print("example.text_b")
                    print(example.text_b)
                    print("word_tokenized_text_b")
                    print(word_tokenized_text_b)
                    print("augmentation_seq_label_mask_word_level")
                    print(augmentation_seq_label_mask_word_level)
                    print("word_idx")
                    print(word_idx)
                    print("tokenized_text_b_word_ids")
                    print(tokenized_text_b_word_ids)
                    print(len(tokenized_text_b_word_ids))
                    print("tokenized_text_b.tokens()")
                    print(tokenized_text_b.tokens())
                    print("tokens_b")
                    print(tokens_b)
                    print(len(tokens_b))
                    raise ValueError
                """
                if example.augmentation_seq_label:
                    augmentation_seq_labels_bpe.append(example.augmentation_seq_label[word_idx])
                else:
                    augmentation_seq_labels_bpe.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                #augmentation_seq_labels_bpe.append(example.augmentation_seq_label[word_idx] if data_args.label_all_tokens else -100)
                augmentation_seq_labels_mask.append(0)
                augmentation_seq_labels_bpe.append(-100)
            previous_word_idx = word_idx
        # append for the final SEP token
        augmentation_seq_labels_mask.append(0)
        augmentation_seq_labels_bpe.append(-100)
        #tokenized_text_word_ids = tokenized_text_word_ids + tokenized_text_b_word_ids + [-100]
        # tokenized_text_word_ids: the corresponding word_id in text_b, others are set to -100

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            augmentation_seq_labels_mask.append(0)
            augmentation_seq_labels_bpe.append(-100)
            #tokenized_text_word_ids.append(-100)
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            augmentation_seq_labels_mask = [0] + augmentation_seq_labels_mask
            augmentation_seq_labels_bpe = [-100] + augmentation_seq_labels_bpe
            #tokenized_text_word_ids = [-100] + tokenized_text_word_ids

        # debug
        if len(tokens) != len(augmentation_seq_labels_bpe):
            print("len(tokens)")
            print(len(tokens))
            print("print(len(augmentation_seq_labels_bpe))")
            print(len(augmentation_seq_labels_bpe))
            print(tokens)
            print(augmentation_seq_labels_bpe)
            raise ValueError

        #print("tokens")
        #print(tokens)
        #print("augmentation_seq_labels_mask")
        #print(augmentation_seq_labels_mask)
        #print("augmentation_seq_labels_bpe")
        #print(augmentation_seq_labels_bpe)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        ####### AUX LOSS DATA
        # get tokens_a mask
        extraction_span_len = len(tokens_a) + 2
        extraction_mask = [1 if 0 < ix < extraction_span_len else 0 for ix in range(max_seq_length)]

        # get extraction labels
        if example.extraction_span:
            ext_start, ext_end = example.extraction_span
            extraction_start_ids = ext_start + 1
            extraction_end_ids = ext_end + 1
        else:
            extraction_start_ids = extraction_span_len
            extraction_end_ids = extraction_span_len

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            augmentation_seq_labels_bpe = ([-100] * padding_length) + augmentation_seq_labels_bpe
            augmentation_seq_labels_mask = ([0] * padding_length) + augmentation_seq_labels_mask
            #tokenized_text_word_ids = ([-100] * padding_length) + tokenized_text_word_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            augmentation_seq_labels_bpe = augmentation_seq_labels_bpe + ([-100] * padding_length)
            augmentation_seq_labels_mask = augmentation_seq_labels_mask + ([0] * padding_length)
            #tokenized_text_word_ids = tokenized_text_word_ids + ([-100] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(augmentation_seq_labels_bpe) == max_seq_length
        assert len(augmentation_seq_labels_mask) == max_seq_length
        #assert len(tokenized_text_word_ids) == max_seq_length

        label_id = multi_label_map_frame_only(example.label)
        """
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "multi_label_classification":
            #label_id = multi_label_map(example.label)
            #label_id = multi_label_map_frame_only(example.label)
            label_id = multi_label_map_frame_only_amr(example.label)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        """

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("ext mask: %s" % " ".join([str(x) for x in extraction_mask]))
            logger.info("ext start: %d" % extraction_start_ids)
            logger.info("ext end: %d" % extraction_end_ids)
            logger.info("aug mask: %s" % " ".join([str(x) for x in augmentation_seq_labels_mask]))
            logger.info("seq labels: %s" % " ".join([str(x) for x in augmentation_seq_labels_bpe]))
            logger.info("word ids: %s" % " ".join([str(x) for x in tokenized_text_b_word_ids]))
            #logger.info("word_ids: %d" % tokenized_text_word_ids)
            if output_mode == "multi_label_classification":
                logger.info("label: %s" % " ".join([str(x) for x in label_id]))
            else:
                logger.info("label: %d" % label_id)

        extraction_start_ids = min(extraction_start_ids, 511)
        extraction_end_ids = min(extraction_end_ids, 511)

        # original tokens b
        if example.original_text_b:
            original_text_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.original_text_b))
            original_text_b_ids = original_text_b_ids[:100]
            if len(original_text_b_ids) < 100:
                original_text_b_ids = original_text_b_ids + [pad_token] * (100 - len(original_text_b_ids))
        else:
            original_text_b_ids = None

        features.append(
            InputSeqLabelFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          extraction_mask=extraction_mask,
                          extraction_start_ids=extraction_start_ids,
                          extraction_end_ids=extraction_end_ids,
                          augmentation_seq_labels_mask=augmentation_seq_labels_mask,
                          augmentation_seq_label=augmentation_seq_labels_bpe,
                          guid=example.guid,
                          original_text_b_ids=original_text_b_ids))
    return features


def convert_binary_seq_label_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True
                                 ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        #print("tokens_a")
        #print(tokens_a)

        tokens_b = None
        """
        print("example")
        print(example.text_a)
        print(example.text_b)
        print(example.label)
        print(example.augmentation_seq_label)
        print(example.augmentation_seq_label_mask)
        print()
        """

        if example.text_b:
            #word_tokenized_text_b = example.text_b.split(" ")
            word_tokenized_text_b = example.text_b
            tokenized_text_b = tokenizer(word_tokenized_text_b, add_special_tokens=False, padding=False, is_split_into_words=True, return_attention_mask=False)
            tokens_b = tokenized_text_b.tokens()
            #print("tokens_b")
            #print(tokens_b)
            #tokens_b = tokenized_text_b["input_ids"]
            #logger.info("token b before trunc")
            #logger.info(" ".join([str(x) for x in tokens_b]))
            # construct seq label mask
            augmentation_seq_label_mask_word_level = example.augmentation_seq_label_mask
            #augmentation_seq_label_mask_word_level = [1] * len(word_tokenized_text_b)
            """
            augmentation_seq_label_mask_word_level = []
            for word in word_tokenized_text_b:
                if word in [")", "("] or word.startswith(":"):
                    augmentation_seq_label_mask_word_level.append(0)
                else:
                    augmentation_seq_label_mask_word_level.append(1)
            """
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            is_tokens_b_truncated = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        #tokenized_text_word_ids = [-100] * len(tokens)
        augmentation_seq_labels_mask = [0] * len(tokens)
        augmentation_seq_labels_bpe = [-100] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            if sep_token_extra:
                segment_ids += [sequence_a_segment_id] * (len(tokens_b) + 1)
            else:
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            #logger.info("token b after trunc")
            #logger.info(" ".join([str(x) for x in tokens_b]))

        tokenized_text_b_word_ids = tokenized_text_b.word_ids()[:len(tokens_b)]
        previous_word_idx = None

        for word_idx in tokenized_text_b_word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                augmentation_seq_labels_mask.append(0)
                augmentation_seq_labels_bpe.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if augmentation_seq_label_mask_word_level:
                    try:
                        augmentation_seq_labels_mask.append(augmentation_seq_label_mask_word_level[word_idx])
                    except:
                        print("example.text_b")
                        print(example.text_b)
                        print("word_tokenized_text_b")
                        print(word_tokenized_text_b)
                        print("augmentation_seq_label_mask_word_level")
                        print(augmentation_seq_label_mask_word_level)
                        print("word_idx")
                        print(word_idx)
                        print("tokenized_text_b_word_ids")
                        print(tokenized_text_b_word_ids)
                        print(len(tokenized_text_b_word_ids))
                        print("tokenized_text_b.tokens()")
                        print(tokenized_text_b.tokens())
                        print("tokens_b")
                        print(tokens_b)
                        print(len(tokens_b))
                        raise ValueError
                else:
                    augmentation_seq_labels_mask.append(1)
                if example.augmentation_seq_label:
                    augmentation_seq_labels_bpe.append(example.augmentation_seq_label[word_idx])
                else:
                    augmentation_seq_labels_bpe.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                #augmentation_seq_labels_bpe.append(example.augmentation_seq_label[word_idx] if data_args.label_all_tokens else -100)
                augmentation_seq_labels_mask.append(0)
                augmentation_seq_labels_bpe.append(-100)
            previous_word_idx = word_idx
        # append for the final SEP token
        augmentation_seq_labels_mask.append(0)
        augmentation_seq_labels_bpe.append(-100)
        #tokenized_text_word_ids = tokenized_text_word_ids + tokenized_text_b_word_ids + [-100]
        # tokenized_text_word_ids: the corresponding word_id in text_b, others are set to -100

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            augmentation_seq_labels_mask.append(0)
            augmentation_seq_labels_bpe.append(-100)
            #tokenized_text_word_ids.append(-100)
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            augmentation_seq_labels_mask = [0] + augmentation_seq_labels_mask
            augmentation_seq_labels_bpe = [-100] + augmentation_seq_labels_bpe
            #tokenized_text_word_ids = [-100] + tokenized_text_word_ids

        # debug
        if len(tokens) != len(augmentation_seq_labels_bpe):
            print("len(tokens)")
            print(len(tokens))
            print("print(len(augmentation_seq_labels_bpe))")
            print(len(augmentation_seq_labels_bpe))
            print(tokens)
            print(augmentation_seq_labels_bpe)
            raise ValueError

        #print("tokens")
        #print(tokens)
        #print("augmentation_seq_labels_mask")
        #print(augmentation_seq_labels_mask)
        #print("augmentation_seq_labels_bpe")
        #print(augmentation_seq_labels_bpe)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        ####### AUX LOSS DATA
        # get tokens_a mask
        extraction_span_len = len(tokens_a) + 2
        extraction_mask = [1 if 0 < ix < extraction_span_len else 0 for ix in range(max_seq_length)]

        # get extraction labels
        if example.extraction_span:
            ext_start, ext_end = example.extraction_span
            extraction_start_ids = ext_start + 1
            extraction_end_ids = ext_end + 1
        else:
            extraction_start_ids = extraction_span_len
            extraction_end_ids = extraction_span_len

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            augmentation_seq_labels_bpe = ([-100] * padding_length) + augmentation_seq_labels_bpe
            augmentation_seq_labels_mask = ([0] * padding_length) + augmentation_seq_labels_mask
            #tokenized_text_word_ids = ([-100] * padding_length) + tokenized_text_word_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            augmentation_seq_labels_bpe = augmentation_seq_labels_bpe + ([-100] * padding_length)
            augmentation_seq_labels_mask = augmentation_seq_labels_mask + ([0] * padding_length)
            #tokenized_text_word_ids = tokenized_text_word_ids + ([-100] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(augmentation_seq_labels_bpe) == max_seq_length
        assert len(augmentation_seq_labels_mask) == max_seq_length
        #assert len(tokenized_text_word_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("ext mask: %s" % " ".join([str(x) for x in extraction_mask]))
            logger.info("ext start: %d" % extraction_start_ids)
            logger.info("ext end: %d" % extraction_end_ids)
            logger.info("aug mask: %s" % " ".join([str(x) for x in augmentation_seq_labels_mask]))
            logger.info("seq labels: %s" % " ".join([str(x) for x in augmentation_seq_labels_bpe]))
            logger.info("word ids: %s" % " ".join([str(x) for x in tokenized_text_b_word_ids]))
            #logger.info("word_ids: %d" % tokenized_text_word_ids)
            logger.info("label: %d" % label_id)

        extraction_start_ids = min(extraction_start_ids, 511)
        extraction_end_ids = min(extraction_end_ids, 511)

        # original tokens b
        if example.original_text_b:
            original_text_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.original_text_b))
            original_text_b_ids = original_text_b_ids[:100]
            if len(original_text_b_ids) < 100:
                original_text_b_ids = original_text_b_ids + [pad_token] * (100 - len(original_text_b_ids))
        else:
            original_text_b_ids = None

        features.append(
            InputSeqLabelFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          extraction_mask=extraction_mask,
                          extraction_start_ids=extraction_start_ids,
                          extraction_end_ids=extraction_end_ids,
                          augmentation_seq_labels_mask=augmentation_seq_labels_mask,
                          augmentation_seq_label=augmentation_seq_labels_bpe,
                          guid=example.guid,
                          original_text_b_ids=original_text_b_ids))
    return features


def multi_label_map(error_name_list):
    label = [0] * 5  # [negative, predicate error, entity error, numerical error + date error, coreference error]
    if "Incorrect" in error_name_list:
        label[0] = 1
    if "PredE" in error_name_list or "RelE" in error_name_list or "NegateSentences" in error_name_list or "neg.verb_antonomy" in error_name_list:
        label[1] = 1
    if "EntE" in error_name_list or "EntitySwap" in error_name_list or "neg.entity" in error_name_list:
        label[2] = 1
    if "CircE" in error_name_list or "DateSwap" in error_name_list or "LocationSwap" in error_name_list:
        label[3] = 1
    if "CorefE" in error_name_list or "PronounSwap" in error_name_list:
        label[4] = 1
    """
    if "Incorrect" in error_name_list:
        label[0] = 1
    if "PredE" in error_name_list or "RelE" in error_name_list:
        label[1] = 1
    if "EntE" in error_name_list:
        label[2] = 1
    if "NumE" in error_name_list or "DateE" in error_name_list or "CircE" in error_name_list:
        label[3] = 1
    if "CorefE" in error_name_list:
        label[4] = 1
    """
    return label


def multi_label_map_frame_only(error_name_list):
    label = [0] * 4  # [negative, predicate error, entity error, numerical error + date error, coreference error]
    if "Incorrect" in error_name_list:
        label[0] = 1
    if "PredE" in error_name_list or "RelE" in error_name_list or "NegateSentences" in error_name_list or "neg.verb_antonomy" in error_name_list:
        label[1] = 1
    if "EntE" in error_name_list or "EntitySwap" in error_name_list or "neg.entity" in error_name_list:
        label[2] = 1
    if "CircE" in error_name_list or "DateSwap" in error_name_list or "LocationSwap" in error_name_list or "neg.date" in error_name_list:
        label[3] = 1
    return label


def multi_label_map_frame_only_amr(error_name_list):
    label = [0] * 4  # [negative, predicate error, entity error, numerical error + date error, coreference error]
    if "Incorrect" in error_name_list:
        label[0] = 1
    if "PredE" in error_name_list or "RelE" in error_name_list or "SentNegatePolarity" in error_name_list or "SentNegateAntonym" in error_name_list:
        label[1] = 1
    if "EntE" in error_name_list or "NERSwap" in error_name_list:
        label[2] = 1
    if "CircE" in error_name_list or "LocationSwap" in error_name_list or "DateEntSwap" in error_name_list:
        label[3] = 1
    return label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    tokens_b_truncated = False
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            tokens_b_truncated = True
    return tokens_b_truncated


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def multi_label_acc_and_f1(preds, labels, prefix=""):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="micro")
    #acc = simple_accuracy(np.reshape(preds, -1), np.reshape(labels, -1))
    #print("multi_label_acc_and_f1")
    return {
        prefix + "acc": acc,
        prefix + "f1": f1
    }


def multi_label_acc_auc_f1(preds, preds_scores, labels, prefix=""):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="micro")
    auc = roc_auc_score(y_true=labels, y_score=preds_scores, average="micro")
    #acc = simple_accuracy(np.reshape(preds, -1), np.reshape(labels, -1))
    #print("multi_label_acc_and_f1")
    return {
        prefix + "acc": acc,
        prefix + "f1": f1,
        prefix + "auc": auc,
    }


def multi_label_bacc_auc_f1(preds, preds_scores, labels, prefix=""):
    n_class = preds.shape[1]
    bacc_scores = []
    for i in range(n_class):
        bacc_scores.append( balanced_accuracy_score(y_true=labels[:, i], y_pred=preds[:, i]) )
    assert len(bacc_scores) == n_class
    bacc_scores = np.asarray(bacc_scores)
    bacc_scores_avg = bacc_scores.mean()
    #acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="micro")
    auc = roc_auc_score(y_true=labels, y_score=preds_scores, average="micro")
    #acc = simple_accuracy(np.reshape(preds, -1), np.reshape(labels, -1))
    #print("multi_label_acc_and_f1")
    return {
        prefix + "bacc": bacc_scores_avg,
        prefix + "f1": f1,
        prefix + "auc": auc,
    }


def multi_label_bacc_auc_f1_macro(preds, preds_scores, labels, prefix=""):
    n_class = preds.shape[1]
    bacc_scores = []
    for i in range(n_class):
        bacc_scores.append( balanced_accuracy_score(y_true=labels[:, i], y_pred=preds[:, i]) )
    assert len(bacc_scores) == n_class
    bacc_scores = np.asarray(bacc_scores)
    bacc_scores_avg = bacc_scores.mean()
    #acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    try:
        auc = roc_auc_score(y_true=labels, y_score=preds_scores, average="macro")
    except:
        print("auc undefined")
        auc = 0.0
    #acc = simple_accuracy(np.reshape(preds, -1), np.reshape(labels, -1))
    #print("multi_label_acc_and_f1")
    return {
        prefix + "bacc": bacc_scores_avg,
        prefix + "f1": f1,
        prefix + "auc": auc,
    }


def multi_label_bacc_f1_macro(preds, labels, prefix=""):
    n_class = preds.shape[1]
    bacc_scores = []
    for i in range(n_class):
        bacc_scores.append( balanced_accuracy_score(y_true=labels[:, i], y_pred=preds[:, i]) )
    assert len(bacc_scores) == n_class
    bacc_scores = np.asarray(bacc_scores)
    bacc_scores_avg = bacc_scores.mean()
    #acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    #acc = simple_accuracy(np.reshape(preds, -1), np.reshape(labels, -1))
    #print("multi_label_acc_and_f1")
    return {
        prefix + "bacc": bacc_scores_avg,
        prefix + "f1": f1
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def complex_metric(preds, labels, prefix=""):
    return {
        prefix + "bacc": balanced_accuracy_score(y_true=labels, y_pred=preds),
        prefix + "f1":   f1_score(y_true=labels, y_pred=preds, average="micro")
    }


def binary_classification_metric(preds, labels, prefix=""):
    return {
        prefix + "bacc": balanced_accuracy_score(y_true=labels, y_pred=preds),
        prefix + "f1": f1_score(y_true=labels, y_pred=preds, average="macro")
    }


def classification_metric_all(preds, preds_scores, labels, prefix=""):
    try:
        auc = roc_auc_score(y_true=labels, y_score=preds_scores, average="macro")
    except:
        print("auc undefined")
        auc = 0.0
    return {
        prefix + "bacc": balanced_accuracy_score(y_true=labels, y_pred=preds),
        prefix + "precision": precision_score(y_true=labels, y_pred=preds),
        prefix + "recall": recall_score(y_true=labels, y_pred=preds),
        prefix + "f1":   f1_score(y_true=labels, y_pred=preds, average="micro"),
        prefix + "auc": auc
    }


def classification_metric_all_macro(preds, preds_scores, labels, prefix=""):
    try:
        auc = roc_auc_score(y_true=labels, y_score=preds_scores, average="macro")
    except:
        print("auc undefined")
        auc = 0.0
    return {
        prefix + "bacc": balanced_accuracy_score(y_true=labels, y_pred=preds),
        prefix + "precision": precision_score(y_true=labels, y_pred=preds),
        prefix + "recall": recall_score(y_true=labels, y_pred=preds),
        prefix + "f1":   f1_score(y_true=labels, y_pred=preds, average="macro"),
        prefix + "auc": auc
    }


def compute_seq_label_metrics(preds, labels, prefix=""):
    label_list = ["True", "False"]
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(preds, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(preds, labels)]
    results = seq_label_metric.compute(predictions=true_predictions, references=true_labels)
    return {prefix + "precision": results["overall_precision"], prefix + "recall": results["overall_recall"], prefix + "f1": results["overall_f1"]}


def compute_metrics(task_name, preds, labels, prefix=""):
    assert len(preds) == len(labels)
    if task_name == "factcc_generated" or task_name == "factcc_binary_label_seq_label_generated":
        return complex_metric(preds, labels, prefix)
    elif task_name == "factcc_annotated":
        return complex_metric(preds, labels, prefix)
    #elif task_name == "factcc_multi_label_generated":
    elif "multi_label" in task_name:
        return multi_label_acc_and_f1(preds, labels, prefix)
    else:
        raise KeyError(task_name)


def merge_arrays(array_list):
    for array in array_list:

        pass
    return

processors = {
    "factcc_generated": FactCCGeneratedProcessor,
    "factcc_annotated": FactCCManualProcessor,
    "factcc_multi_label_generated": FactCCMultiLabelGeneratedProcessor,
    "factcc_amr_multi_label_generated": FactCCAmrMultiLabelGeneratedProcessor,
    "factcc_amr_aug_multi_label_generated": FactCCAmrAugMultiLabelGeneratedProcessor,
    "factcc_amr_aug_multi_label_seq_label_generated": FactCCAmrAugMultiLabelSeqLabelGeneratedProcessor,
    "factcc_multi_label_seq_label_generated": FactCCMultiLabelSeqLabelGeneratedProcessor,
    "factcc_multi_label_attn_generated": FactCCMultiLabelAttnGeneratedProcessor,
    "frank_multi_label_annotated": FrankMultiLabelManualProcessor,
    "frank_multi_label_annotated_filtered": FrankMultiLabelFilteredManualProcessor,
    "frank_multi_label_srl_annotated": FrankMultiLabelSRLManualProcessor,
    "frank_multi_label_srl_nested_annotated": FrankMultiLabelSRLNestedManualProcessor,
    "frank_amr_multi_label_annotated": FrankAmrMultiLabelManualProcessor,
    "frank_amr_multi_label_seq_label_annotated": FrankAmrMultiLabelSeqLabelManualProcessor,
    "frank_multi_label_srl_nested_seq_label_annotated": FrankSRLnestedMultiLabelSeqLabelManualProcessor,
    "frank_multi_label_attn_annotated": FrankMultiLabelAttnManualProcessor,
    "aggrefact_multi_label_annotated": AggreFactMultiLabelAnnotatedProcessor,
    "aggrefact_multi_label_annotated_with_entire_sent": AggreFactMultiLabelAnnotatedWithEntireSentProcessor,
    "aggrefact_multi_label_annotated_with_entire_sent_four_label": AggreFactMultiLabelAnnotatedWithEntireSentFourLabelProcessor,
    "aggrefact_multi_label_synthetic": AggreFactMultiLabelSyntheticProcessor,
    "aggrefact_multi_label_claim_attn_synthetic": AggreFactMultiLabelClaimAttnSyntheticProcessor,
    "aggrefact_multi_label_claim_attn_annotated": AggreFactMultiLabelClaimAttnAnnotatedProcessor,
    "aggrefact_multi_label_claim_attn_annotated_with_entire_sent": AggreFactMultiLabelClaimAttnAnnotatedWithEntireSentProcessor,
    "aggrefact_multi_label_claim_attn_annotated_with_entire_sent_four_label": AggreFactMultiLabelClaimAttnAnnotatedWithEntireSentFourLabelProcessor,
    "factcc_binary_label_seq_label_generated": FactCCBinarySeqLabelGeneratedProcessor,
}

output_modes = {
    "factcc_generated": "classification",
    "factcc_annotated": "classification",
    "factcc_multi_label_generated": "multi_label_classification",
    "factcc_amr_multi_label_generated": "multi_label_classification",
    "factcc_amr_aug_multi_label_generated": "multi_label_classification",
    "factcc_amr_aug_multi_label_seq_label_generated": "multi_label_classification",
    "factcc_multi_label_seq_label_generated": "multi_label_classification",
    "factcc_binary_label_seq_label_generated": "classification",
    "frank_multi_label_annotated": "multi_label_classification",
    "frank_multi_label_annotated_filtered": "multi_label_classification",
    "frank_multi_label_srl_annotated": "multi_label_classification",
    "frank_multi_label_srl_nested_annotated": "multi_label_classification",
    "frank_amr_multi_label_annotated": "multi_label_classification",
    "frank_amr_multi_label_seq_label_annotated": "multi_label_classification",
    "frank_multi_label_srl_nested_seq_label_annotated": "multi_label_classification",
    "factcc_multi_label_attn_generated": "multi_label_classification",
    "frank_multi_label_attn_annotated": "multi_label_classification",
    "aggrefact_multi_label_annotated": "multi_label_classification",
    "aggrefact_multi_label_synthetic": "multi_label_classification",
    "aggrefact_multi_label_claim_attn_synthetic": "multi_label_classification",
    "aggrefact_multi_label_claim_attn_annotated": "multi_label_classification",
    "aggrefact_multi_label_annotated_with_entire_sent": "multi_label_classification",
    "aggrefact_multi_label_annotated_with_entire_sent_four_label": "multi_label_classification",
    "aggrefact_multi_label_claim_attn_annotated_with_entire_sent": "multi_label_classification",
    "aggrefact_multi_label_claim_attn_annotated_with_entire_sent_four_label": "multi_label_classification"
}

GLUE_TASKS_NUM_LABELS = {
    "factcc_generated": 2,
    "factcc_annotated": 2,
}
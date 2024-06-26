"""
Updated dataset classes for language modeling.
Also includes the data collator.
Uses a new IterableTextDataset class, allowing iterable training data, to
avoid storing the entire training dataset in memory.
"""

import codecs
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Union, Tuple

from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset, IterableDataset
from filelock import FileLock
from transformers import PreTrainedTokenizer, BatchEncoding
from tqdm import tqdm


# Text dataset that iterates through a file.
# Iterable file can be obtained by using the saved examples generated by
# LineByLineTextDataset, converting into a text file.
#
# If run_transformer_language_modeling.py is run with python -m torch.distributed.launch --nproc_per_node N_GPUS,
# then the different devices are automatically handled. One iterable is created
# for each device, but the iterables are coordinated.
# See use DistributedSampler in Hugging Face trainer.py.
# This means that the iterable does not need to account for multiple workers, so we
# can ignore: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
class IterableTextDataset(IterableDataset):
    """
    Iterable version of TextDataset. Data must be preshuffled.
    Each line in the input file should be a string of integers separated by spaces.
    [CLS] and [SEP] tokens should already be included.
    Each line should correspond to one example (one or two sentences), but
    padding and truncation is handled automatically.
    """

    # This is the iterator that is returned by the iter() method.
    class ExampleIterator:
        def __init__(self, file_path: str, block_size: int,
                     pad_token_id: int, sep_token_id: int):
            self.file_path = file_path
            self.block_size = block_size
            self.pad_token_id = pad_token_id
            self.sep_token_id = sep_token_id
            # Start the input file from the beginning.
            self.input_file = codecs.open(file_path, 'rb', encoding='utf-8')

        def __iter__(self):
            return self

        def __next__(self):
            return self._next_example()

        # Get one example at a time.
        def _next_example(self):
            # In case this is called after the input file was closed.
            if self.input_file == None:
                print('No input file (or input file has been closed).')
                raise StopIteration
            # Try to read a string of space-separated integers.
            # Each example should be: [CLS] sent_1 [SEP] sent_2 [SEP]
            example_string = self.input_file.readline()
            while example_string == '\n': # Skip new lines.
                example_string = self.input_file.readline()
            if example_string == '': # This only occurs at the end of a file (otherwise there would at least be a newline character).
                self.input_file.close()
                print('Example iterator complete.')
                self.input_file = None
                raise StopIteration
            # Process example.
            example_string = example_string.strip()
            example = [int(token_id) for token_id in example_string.split()]
            # Truncating is done here.
            if len(example) > self.block_size:
                example = example[0:self.block_size]
            # Padding is handled by the collator.
            return { "input_ids": torch.tensor(example, dtype=torch.long) }

    # Init IterableTextDataset.
    def __init__(
        self,
        file_path: str,
        block_size: int,
        pad_token_id: int,
        sep_token_id: int,
        n_examples: int = -1,
    ):
        super(IterableTextDataset).__init__()
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.input_filepath = file_path
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        # Initially get total num_examples.
        if n_examples > 0:
            self.num_examples = n_examples
        else:
            print("Counting examples in train file. This can be slow.")
            example_count = 0
            infile = codecs.open(file_path, 'rb', encoding='utf-8')
            for line in tqdm(infile):
                example_count += 1
            infile.close()
            self.num_examples = example_count
            print("Finished counting: {} examples.".format(example_count))

    def __iter__(self):
        return self.ExampleIterator(self.input_filepath, self.block_size,
                                    self.pad_token_id, self.sep_token_id)

    def __len__(self):
        return self.num_examples


# Edited to process incrementally, combine every two sentences, and cache the created examples.
# The cached examples are unpadded and untruncated. They are padded/truncated on the fly.
class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_lm_line_pairs_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.block_size = block_size

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            start_line = 0
            max_stored_line_count = 100000 # Ideally, should be an even number so that sentences are paired together.
            save_every = 1000000 # Save a copy of the examples every n lines.

            print("Creating features from dataset file at {}".format(directory))

            # Load existing features file.
            self.examples = []
            if start_line != 0:
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)

            textfile = open(file_path, encoding="utf-8")
            total_line_count = 0
            stored_lines = []
            for line in textfile:
                total_line_count += 1
                if total_line_count <= start_line:
                    continue
                stripped_line = line.strip()
                if stripped_line != '':
                    stored_lines.append(stripped_line)
                # Process the currently stored lines.
                if total_line_count % max_stored_line_count == 0:
                    print("Processing {} lines.".format(max_stored_line_count))
                    batch_encoding = tokenizer(stored_lines, add_special_tokens=False, truncation=True, max_length=99999)
                    for i in range(0, len(batch_encoding["input_ids"])-1, 2): # Subtract one from the length so that the last unpaired example is skipped.
                        sent_1 = batch_encoding["input_ids"][i]
                        sent_2 = batch_encoding["input_ids"][i+1]
                        if self.cls_token_id is not None and self.sep_token_id is not None:
                            example = [self.cls_token_id] + sent_1 +  [self.sep_token_id] + sent_2 + [self.sep_token_id]
                        else:
                            # Some pre-trained tokenizers (e.g. GPT2) have no [SEP] or [CLS] tokens.
                            example = sent_1 + sent_2
                        # Note that these examples are unpadded and un-truncated.
                        self.examples.append(example)
                    stored_lines = []
                if total_line_count % save_every == 0:
                    # Save a copy of the examples so far.
                    print("Saving a copy of {} examples.".format(len(self.examples)))
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            textfile.close()
            # Process the remaining set of lines. This is copied from above for maximal bad code style!
            batch_encoding = tokenizer(stored_lines, add_special_tokens=False, truncation=True, max_length=99999)
            for i in range(0, len(batch_encoding["input_ids"])-1, 2):
                sent_1 = batch_encoding["input_ids"][i]
                sent_2 = batch_encoding["input_ids"][i+1]
                if self.cls_token_id is not None and self.sep_token_id is not None:
                    example = [self.cls_token_id] + sent_1 +  [self.sep_token_id] + sent_2 + [self.sep_token_id]
                else:
                    example = sent_1 + sent_2
                self.examples.append(example)

            # Cached, but this is just in case it is needed later (e.g. to convert the cached version into an iterable).
            print("Saving a copy of {} examples.".format(len(self.examples)))
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        # Truncate or pad on the fly.
        example = self.examples[i]
        if len(example) > self.block_size:
            example = example[0:self.block_size]
        # Padding is handled by the collator.
        return { "input_ids": torch.tensor(example, dtype=torch.long) }


# Data collator for language modeling.
# Includes the attention mask based on pad tokens.
@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
        else:
            # Note: for GPT-2, the inputs/labels are automatically shifted
            # inside the model for autoregressive language modeling.
            inputs = batch
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = inputs != self.tokenizer.pad_token_id
        if self.tokenizer.pad_token_id is None:
            # Replace placeholder pad tokens, which are masked out anyways.
            attention_mask = inputs != -1
            labels[labels == -1] = -100
            inputs[inputs == -1] = 1 # Default to token 1, but will be masked out.
        return {"input_ids": inputs, "attention_mask": attention_mask, "labels": labels}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                # Use -1 as a placeholder pad token.
                return pad_sequence(examples, batch_first=True, padding_value=-1)
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is None:
            # Placeholder pad token.
            padding_mask = labels.eq(-1)
        else:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

"""Dataset loading and train/validation/test splitting utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import nltk
from nltk.corpus import treebank

Sentence = List[Tuple[str, str]]

NLTK_RESOURCES = {"corpora/treebank": "treebank"}


def ensure_nltk_data() -> None:
    """Download the Penn Treebank sample if it is not already available."""
    for resource, package in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def load_treebank() -> List[Sentence]:
    """Load the Penn Treebank sample as a list of (word, tag) sentences."""
    ensure_nltk_data()
    return [list(sentence) for sentence in treebank.tagged_sents()]


@dataclass
class DataSplit:
    train: List[Sentence]
    val: List[Sentence]
    test: List[Sentence]
    seed: int = 42

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "train_sentences": len(self.train),
            "val_sentences": len(self.val),
            "test_sentences": len(self.test),
            "train_tokens": count_tokens(self.train),
            "val_tokens": count_tokens(self.val),
            "test_tokens": count_tokens(self.test),
        }


def split_sentences(
    sentences: Sequence[Sentence],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> DataSplit:
    """Split sentences deterministically into train/validation/test partitions."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1")

    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)

    n_total = len(order)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    return DataSplit(
        train=[sentences[i] for i in train_idx],
        val=[sentences[i] for i in val_idx],
        test=[sentences[i] for i in test_idx],
        seed=seed,
    )


def count_tokens(sentences: Sequence[Sentence]) -> int:
    return sum(len(sentence) for sentence in sentences)


def tokens_of(sentences: Sequence[Sentence]) -> List[str]:
    return [word for sentence in sentences for word, _ in sentence]


def tags_of(sentences: Sequence[Sentence]) -> List[str]:
    return [tag for sentence in sentences for _, tag in sentence]


def vocabulary(sentences: Sequence[Sentence]) -> Set[str]:
    return {word for sentence in sentences for word, _ in sentence}


def tagset(sentences: Sequence[Sentence]) -> List[str]:
    return sorted({tag for sentence in sentences for _, tag in sentence})


def oov_words(train: Sequence[Sentence], test: Sequence[Sentence]) -> Set[str]:
    """Test-set word types that never occur in the training split."""
    return vocabulary(test) - vocabulary(train)

"""Statistical POS taggers: a bigram HMM with Viterbi decoding and N-gram backoff."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from nltk.tag import BigramTagger, DefaultTagger, TrigramTagger, UnigramTagger

from .data import Sentence

UNK = "<UNK>"
BOS = "<s>"


class HMMTagger:
    """Bigram HMM with add-alpha smoothing and suffix-based unknown-word emissions."""

    name = "Statistical (HMM)"

    def __init__(
        self,
        alpha: float = 0.1,
        min_word_freq: int = 2,
        suffix_weight: float = 0.35,
    ) -> None:
        self.alpha = alpha
        self.min_word_freq = min_word_freq
        self.suffix_weight = suffix_weight

    def fit(self, train: Sequence[Sentence]) -> "HMMTagger":
        word_freq = Counter(word for sentence in train for word, _ in sentence)
        self.word_vocab = {
            word for word, count in word_freq.items() if count >= self.min_word_freq
        }
        self.tags = sorted({tag for sentence in train for _, tag in sentence})
        self.tag_index = {tag: i for i, tag in enumerate(self.tags)}
        n_tags = len(self.tags)

        start_counts: Counter = Counter()
        trans_counts: Counter = Counter()
        emit_counts: Counter = Counter()
        tag_token_totals: Counter = Counter()
        suffix_counts: Counter = Counter()
        suffix_totals: Counter = Counter()

        for sentence in train:
            previous = BOS
            for position, (word, tag) in enumerate(sentence):
                known = word in self.word_vocab
                emit_word = word if known else UNK
                if position == 0:
                    start_counts[tag] += 1
                trans_counts[(previous, tag)] += 1
                emit_counts[(tag, emit_word)] += 1
                tag_token_totals[tag] += 1

                suffix = word[-3:].lower()
                suffix_counts[(tag, suffix)] += 1
                suffix_totals[suffix] += 1
                previous = tag

        self.start_counts = start_counts
        self.trans_counts = trans_counts
        self.emit_counts = emit_counts
        self.tag_token_totals = tag_token_totals
        self.suffix_counts = suffix_counts
        self.suffix_totals = suffix_totals
        self.n_tags = n_tags
        self.n_sentences = len(train)
        self.n_tokens = sum(tag_token_totals.values())
        self.vocab_size = len(self.word_vocab) + 1

        self._start_log = np.array(
            [
                math.log(
                    (start_counts[tag] + self.alpha)
                    / (self.n_sentences + self.alpha * n_tags)
                )
                for tag in self.tags
            ]
        )

        self._trans_log = np.empty((n_tags, n_tags))
        for previous in self.tags:
            row_total = sum(
                trans_counts[(previous, tag)] for tag in self.tags
            )
            for tag in self.tags:
                count = trans_counts[(previous, tag)]
                probability = (count + self.alpha) / (
                    row_total + self.alpha * n_tags
                )
                self._trans_log[self.tag_index[previous], self.tag_index[tag]] = (
                    math.log(probability)
                )

        self._tag_prior = np.array(
            [
                (tag_token_totals[tag] + self.alpha)
                / (self.n_tokens + self.alpha * n_tags)
                for tag in self.tags
            ]
        )
        return self

    def _emission_scores(self, word: str) -> np.ndarray:
        scores = np.empty(self.n_tags)
        is_unknown = word not in self.word_vocab
        suffix = word[-3:].lower()
        suffix_seen = self.suffix_totals.get(suffix, 0) >= 3

        for i, tag in enumerate(self.tags):
            total = self.tag_token_totals[tag]
            denominator = total + self.alpha * self.vocab_size
            unknown_probability = (
                self.emit_counts.get((tag, UNK), 0) + self.alpha
            ) / denominator

            if not is_unknown:
                probability = (
                    self.emit_counts.get((tag, word), 0) + self.alpha
                ) / denominator
            elif suffix_seen:
                suffix_probability = (
                    self.suffix_counts.get((tag, suffix), 0) + self.alpha
                ) / (self.suffix_totals[suffix] + self.alpha * self.n_tags)
                probability = (
                    (1.0 - self.suffix_weight) * unknown_probability
                    + self.suffix_weight * suffix_probability
                )
            else:
                probability = (
                    (1.0 - self.suffix_weight) * unknown_probability
                    + self.suffix_weight * self._tag_prior[i]
                )
            scores[i] = math.log(max(probability, 1e-300))
        return scores

    def tag(self, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []

        scores = self._start_log + self._emission_scores(tokens[0])
        backpointers: List[np.ndarray] = []

        for token in tokens[1:]:
            candidates = scores[:, None] + self._trans_log
            best_previous = candidates.argmax(axis=0)
            scores = (
                candidates[best_previous, np.arange(self.n_tags)]
                + self._emission_scores(token)
            )
            backpointers.append(best_previous)

        best_last = int(scores.argmax())
        path = [best_last]
        for backpointer in reversed(backpointers):
            path.append(int(backpointer[path[-1]]))
        path.reverse()
        return [self.tags[index] for index in path]

    def top_transitions(self, k: int = 10) -> List[dict]:
        rows = []
        for (previous, tag), count in self.trans_counts.most_common(k):
            probability = math.exp(
                self._trans_log[self.tag_index[previous], self.tag_index[tag]]
            )
            rows.append(
                {
                    "previous_tag": previous,
                    "tag": tag,
                    "count": count,
                    "probability": round(probability, 4),
                }
            )
        return rows

    def top_emissions(self, k: int = 10) -> List[dict]:
        rows = []
        for (tag, word), count in self.emit_counts.most_common(k):
            rows.append({"tag": tag, "word": word, "count": count})
        return rows


class NgramBackoffTagger:
    """Trigram tagger backing off to bigram, unigram, then the default tag."""

    name = "Statistical (N-gram Backoff)"

    def __init__(self, default_tag: str = "NN") -> None:
        self.default_tag = default_tag
        self._tagger = None

    def fit(self, train: Sequence[Sentence]) -> "NgramBackoffTagger":
        train = list(train)
        unigram = UnigramTagger(
            train, backoff=DefaultTagger(self.default_tag), cutoff=1
        )
        bigram = BigramTagger(train, backoff=unigram, cutoff=1)
        self._tagger = TrigramTagger(train, backoff=bigram, cutoff=1)
        return self

    def tag(self, tokens: Sequence[str]) -> List[str]:
        if self._tagger is None:
            raise RuntimeError("Call fit() before tag()")
        return [tag for _, tag in self._tagger.tag(list(tokens))]

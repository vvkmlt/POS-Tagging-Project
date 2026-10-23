"""Evaluation utilities shared by all three tagging approaches."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import Sentence

TagFunction = Callable[[Sequence[str]], List[str]]


@dataclass
class EvaluationResult:
    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    oov_accuracy: float
    n_tokens: int
    n_oov: int
    seconds: float
    tokens_per_second: float
    per_tag_f1: Dict[str, float] = field(default_factory=dict)
    per_tag_support: Dict[str, int] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    confusion: List[List[int]] = field(default_factory=list)
    gold: List[str] = field(default_factory=list, repr=False)
    predictions: List[str] = field(default_factory=list, repr=False)
    tokens: List[str] = field(default_factory=list, repr=False)

    def metrics(self) -> dict:
        return {
            "Model": self.name,
            "Accuracy": self.accuracy,
            "Macro F1": self.macro_f1,
            "Weighted F1": self.weighted_f1,
            "OOV Accuracy": self.oov_accuracy,
            "Tokens": self.n_tokens,
            "OOV Tokens": self.n_oov,
            "Inference Seconds": self.seconds,
            "Tokens/Second": self.tokens_per_second,
        }


def evaluate_tagger(
    name: str,
    tag_function: TagFunction,
    test_sentences: Sequence[Sentence],
    oov_words: Set[str],
) -> EvaluationResult:
    tokens: List[str] = []
    gold: List[str] = []
    predictions: List[str] = []

    start = time.perf_counter()
    for sentence in test_sentences:
        sentence_tokens = [word for word, _ in sentence]
        sentence_gold = [tag for _, tag in sentence]
        sentence_predictions = tag_function(sentence_tokens)
        if len(sentence_predictions) != len(sentence_gold):
            raise ValueError(
                f"{name} returned {len(sentence_predictions)} tags for a sentence "
                f"with {len(sentence_gold)} tokens"
            )
        tokens.extend(sentence_tokens)
        gold.extend(sentence_gold)
        predictions.extend(sentence_predictions)
    seconds = time.perf_counter() - start

    gold_array = np.array(gold)
    prediction_array = np.array(predictions)
    oov_mask = np.array([token in oov_words for token in tokens])

    labels = sorted(set(gold) | set(predictions))
    report = classification_report(
        gold,
        predictions,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    per_tag_f1 = {
        tag: float(values["f1-score"])
        for tag, values in report.items()
        if isinstance(values, dict) and tag in labels
    }
    per_tag_support = {
        tag: int(values["support"])
        for tag, values in report.items()
        if isinstance(values, dict) and tag in labels
    }

    return EvaluationResult(
        name=name,
        accuracy=float(accuracy_score(gold, predictions)),
        macro_f1=float(f1_score(gold, predictions, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(gold, predictions, average="weighted", zero_division=0)
        ),
        oov_accuracy=(
            float(accuracy_score(gold_array[oov_mask], prediction_array[oov_mask]))
            if oov_mask.any()
            else float("nan")
        ),
        n_tokens=len(tokens),
        n_oov=int(oov_mask.sum()),
        seconds=seconds,
        tokens_per_second=len(tokens) / max(seconds, 1e-9),
        per_tag_f1=per_tag_f1,
        per_tag_support=per_tag_support,
        labels=labels,
        confusion=confusion_matrix(gold, predictions, labels=labels).tolist(),
        gold=gold,
        predictions=predictions,
        tokens=tokens,
    )


def results_frame(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    frame = pd.DataFrame([result.metrics() for result in results])
    numeric = ["Accuracy", "Macro F1", "Weighted F1", "OOV Accuracy"]
    frame[numeric] = frame[numeric].round(4)
    frame["Inference Seconds"] = frame["Inference Seconds"].round(3)
    frame["Tokens/Second"] = frame["Tokens/Second"].round(1)
    return frame


def per_tag_frame(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {result.name: result.per_tag_f1 for result in results}
    ).fillna(0.0)
    supports = pd.DataFrame(
        {result.name: result.per_tag_support for result in results}
    )
    frame["Support"] = supports.iloc[:, 0].fillna(0).astype(int)
    frame.index.name = "Tag"
    return frame.sort_values("Support", ascending=False).round(4)


def error_examples(result: EvaluationResult, limit: int = 12) -> List[dict]:
    examples = []
    for token, gold, prediction in zip(
        result.tokens, result.gold, result.predictions
    ):
        if gold != prediction:
            examples.append(
                {"Token": token, "Gold": gold, f"{result.name} Prediction": prediction}
            )
        if len(examples) >= limit:
            break
    return examples

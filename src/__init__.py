"""Source package for the POS tagging comparison project."""

from .data import DataSplit, load_treebank, oov_words, split_sentences
from .evaluate import EvaluationResult, evaluate_tagger, results_frame
from .neural import NeuralTagger, train_bilstm
from .rule_based import BrillRuleTagger, RegexRuleTagger
from .statistical import HMMTagger, NgramBackoffTagger

__all__ = [
    "DataSplit",
    "BrillRuleTagger",
    "EvaluationResult",
    "HMMTagger",
    "NeuralTagger",
    "NgramBackoffTagger",
    "RegexRuleTagger",
    "evaluate_tagger",
    "load_treebank",
    "oov_words",
    "results_frame",
    "split_sentences",
    "train_bilstm",
]

"""Generate the project notebook from a sequence of markdown and code cells."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "POS_Tagging_Comparison.ipynb"


def md(source: str):
    return new_markdown_cell(source.strip("\n"))


def code(source: str):
    return new_code_cell(source.strip("\n"))


cells = []

cells.append(
    md(
        r"""
# A Theoretical and Empirical Comparison of Rule-Based, Statistical, and Neural POS Tagging

**Mini Project / Term Project**

- **Author:** VENKATESH V
- **Roll Number:** E720524AM034
- **Course:** B.E - CSE (AI & ML) — Course Code U24DSP13
- **Institution:** EASA COLLEGE OF ENGINEERING AND TECHNOLOGY
- **Date:** October 2026
- **Repository:** https://github.com/vvkmlt/POS-Tagging-Project

---

This notebook implements and benchmarks three families of Part-of-Speech (POS) taggers on the
Penn Treebank sample distributed with NLTK:

| Family | Implementation | Learning signal |
|---|---|---|
| Rule-based | Hand-written regex/lexicon tagger | Human linguistic knowledge |
| Rule-based (learned rules) | Brill transformation-based learning | Error-driven rule induction |
| Statistical | Bigram HMM with Viterbi decoding | Transition and emission counts |
| Statistical | Trigram tagger with back-off | N-gram counts |
| Neural | Character-aware BiLSTM | Learned distributed representations |

All models are trained on the same split and evaluated on the same held-out test sentences.
"""
    )
)

cells.append(
    md(
        r"""
## 1. Introduction and Objectives

POS tagging assigns a grammatical category (noun, verb, determiner, ...) to every token in a
sentence. It is a foundational sequence-labelling task that feeds parsing, named-entity
recognition, machine translation, and information extraction.

The project answers three questions:

1. **How large is the accuracy gap** between hand-crafted rules, probabilistic sequence models,
   and neural sequence models on the same corpus?
2. **How do the approaches behave on out-of-vocabulary (OOV) words**, where memorisation and
   hand-written rules are weakest?
3. **What are the engineering trade-offs** in training cost, inference speed, and
   interpretability?

### Deliverables covered by this notebook

- Complete, reproducible Jupyter notebook with all experiments.
- Tables, figures, error analyses, and textual discussion for the PDF report.
- Saved artifacts (`reports/results.json`, `reports/model_comparison.csv`, `assets/figures/*.png`)
  used to build the report and the screenshot PDF.
"""
    )
)

cells.append(
    md(
        r"""
## 2. Experimental Setup

The following choices keep the comparison fair:

- **Corpus:** `nltk.corpus.treebank` (Penn Treebank sample, 3,914 sentences, PTB tagset).
- **Split:** sentence-level 80/10/10 train/validation/test with a fixed seed (42). Validation data
  is used only for neural early stopping; it is never used to fit the rule-based or statistical
  taggers.
- **Evaluation:** every tagger receives raw token sequences and must return one tag per token.
- **Metrics:** token accuracy, macro F1, weighted F1, OOV accuracy, inference throughput, and
  training time.
- **Hardware:** timings are measured on the CPU of the machine running this notebook.
"""
    )
)

cells.append(
    code(
        r"""
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src" / "data.py").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the POS tagging project repository.")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "assets" / "figures"
REPORT_DIR = ROOT / "reports"
for directory in (FIG_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

from src.data import (
    count_tokens,
    load_treebank,
    oov_words,
    split_sentences,
    tagset,
    tags_of,
    tokens_of,
)
from src.evaluate import (
    error_examples,
    evaluate_tagger,
    per_tag_frame,
    results_frame,
)
from src.neural import NeuralTagger, build_vocabs, train_bilstm
from src.rule_based import REGEX_RULES, BrillRuleTagger, RegexRuleTagger
from src.statistical import HMMTagger, NgramBackoffTagger

print("Project root:", ROOT)
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Device: CPU")
print("Processor:", platform.processor() or platform.machine())
"""
    )
)

cells.append(
    md(
        r"""
## 3. Dataset Preparation and Exploratory Data Analysis

The Penn Treebank sample is already tokenised and tagged with the 45-tag PTB tagset. We keep the
original tagset so that the results are directly comparable across all five models.

The split is performed at the **sentence** level to avoid leaking fragments of the same sentence
into training and testing. The test partition is used once, after all models are fitted.
"""
    )
)

cells.append(
    code(
        r"""
sentences = load_treebank()
split = split_sentences(sentences, train_frac=0.8, val_frac=0.1, seed=SEED)

summary = pd.DataFrame(split.summary().items(), columns=["Property", "Value"]).set_index("Property")
display(summary)

train_tags = tagset(split.train)
test_tags = tagset(split.test)
print(f"Tags observed in training: {len(train_tags)}")
print(f"Tags observed in test:     {len(test_tags)}")
print("Training tagset:", ", ".join(train_tags))
"""
    )
)

cells.append(
    code(
        r"""
all_tagged = split.train + split.val + split.test
tag_counts = pd.Series(tags_of(all_tagged)).value_counts()
tag_table = pd.DataFrame(
    {
        "Train": pd.Series(tags_of(split.train)).value_counts(),
        "Validation": pd.Series(tags_of(split.val)).value_counts(),
        "Test": pd.Series(tags_of(split.test)).value_counts(),
    }
).fillna(0).astype(int)
tag_table["Share of Test Tokens"] = (tag_table["Test"] / tag_table["Test"].sum()).round(4)
tag_table.index.name = "Tag"
display(tag_table.head(20))

fig, ax = plt.subplots(figsize=(13, 4.5))
tag_table["Train"].plot.bar(ax=ax, color=COLORS[0], width=0.8)
ax.set_title("Training-set tag distribution (Penn Treebank sample)")
ax.set_xlabel("PTB tag")
ax.set_ylabel("Token count")
ax.tick_params(axis="x", rotation=90, labelsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_tag_distribution.png")
plt.show()

print(f"Most frequent tag: {tag_counts.index[0]} ({tag_counts.iloc[0]} tokens)")
print(f"Rarest tag:       {tag_counts.index[-1]} ({tag_counts.iloc[-1]} tokens)")
"""
    )
)

cells.append(
    code(
        r"""
sentence_lengths = np.array([len(sentence) for sentence in all_tagged])
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(sentence_lengths, bins=40, color=COLORS[2], edgecolor="white")
ax.set_title("Sentence length distribution")
ax.set_xlabel("Tokens per sentence")
ax.set_ylabel("Number of sentences")
ax.axvline(sentence_lengths.mean(), color="black", linestyle="--", label=f"Mean = {sentence_lengths.mean():.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_sentence_lengths.png")
plt.show()

print(f"Sentences: {len(all_tagged):,}")
print(f"Tokens:    {len(tokens_of(all_tagged)):,}")
print(f"Longest sentence: {sentence_lengths.max()} tokens")
"""
    )
)

cells.append(
    code(
        r"""
oov = oov_words(split.train, split.test)
test_tokens = tokens_of(split.test)
test_oov_mask = np.array([token in oov for token in test_tokens])

oov_frame = pd.DataFrame(
    {
        "Quantity": [
            "Test tokens",
            "OOV tokens",
            "OOV rate (tokens)",
            "Distinct test word types",
            "Distinct OOV word types",
            "OOV rate (types)",
        ],
        "Value": [
            len(test_tokens),
            int(test_oov_mask.sum()),
            f"{test_oov_mask.mean():.2%}",
            len(set(test_tokens)),
            len(oov),
            f"{len(oov) / len(set(test_tokens)):.2%}",
        ],
    }
).set_index("Quantity")
display(oov_frame)

sample_oov = pd.Series(sorted(oov)).sample(15, random_state=SEED)
print("Sample OOV words:", ", ".join(sample_oov))
"""
    )
)

cells.append(
    code(
        r"""
pd.set_option("display.max_colwidth", 140)
sample_row = pd.DataFrame(
    [
        {
            "Sentence (tokens)": " ".join(word for word, _ in sentence),
            "Gold tags": " ".join(tag for _, tag in sentence),
        }
        for sentence in split.test[:5]
    ]
)
display(sample_row)
pd.reset_option("display.max_colwidth")
"""
    )
)

cells.append(
    md(
        r"""
### Evaluation protocol

For each model we compute:

- **Accuracy** = correctly tagged tokens / all test tokens.
- **Macro F1** = unweighted mean of per-tag F1, which exposes weakness on rare tags.
- **Weighted F1** = support-weighted mean of per-tag F1 (comparable to accuracy but penalising
  systematic confusions).
- **OOV accuracy** = accuracy restricted to test tokens that never occurred in training.
- **Inference throughput** = test tokens per second, tagging one sentence at a time on CPU.
- **Training time** = wall-clock time to fit the model (zero for the hand-written tagger).

Training time is measured separately from inference time; neural inference includes the PyTorch
forward pass and tensor construction per sentence, which is the realistic deployment API here.
"""
    )
)

cells.append(
    md(
        r"""
## 4. Approach 1: Rule-Based Tagging

### 4.1 Hand-written regex and lexicon rules

A rule-based tagger encodes linguistic knowledge directly. We use an ordered cascade:

1. Punctuation and symbol lookup.
2. Irregular-form lexicon (e.g., *was* -> `VBD`, *children* -> `NNS`).
3. Closed-class word lists (pronouns, determiners, prepositions, modals, ...).
4. Morphological regex rules (e.g., `-ing` -> `VBG`, `-ed` -> `VBD`, `-ly` -> `RB`,
   capitalised token -> `NNP`, trailing `-s` -> `NNS`).
5. Default tag `NN` when no rule fires.

The cascade is ordered from most to least specific. This model has **no learned parameters**, so
its training time is zero, and every decision can be inspected by a human.

**Limitations:** English morphology and syntax are highly ambiguous. A single suffix is not a
reliable tag predictor (*running* can be `VBG` or `NN`; *fast* can be `JJ`, `RB`, or `NN`), and
hand-written rules cannot resolve context-dependent ambiguity.
"""
    )
)

cells.append(
    code(
        r"""
regex_tagger = RegexRuleTagger()

rule_table = pd.DataFrame(REGEX_RULES, columns=["Regex pattern", "Assigned tag"])
display(rule_table)

demo_tokens = "The quick brown foxes were running through the old houses .".split()
demo_frame = pd.DataFrame(
    {
        "Token": demo_tokens,
        "Predicted tag": regex_tagger.tag(demo_tokens),
    }
)
display(demo_frame)

start = time.perf_counter()
regex_result = evaluate_tagger(regex_tagger.name, regex_tagger.tag, split.test, oov)
regex_train_seconds = 0.0
regex_eval_seconds = time.perf_counter() - start

display(pd.DataFrame([regex_result.metrics()]).round(4))
print(classification_report(regex_result.gold, regex_result.predictions, zero_division=0))
"""
    )
)

cells.append(
    md(
        r"""
### 4.2 Brill transformation-based learning (TBL)

Brill tagging still produces a symbolic, human-readable rule set, but the rules are **learned from
data** rather than written by hand.

1. Start with a unigram tagger as the baseline (most frequent tag per word).
2. Compare the baseline output with the gold tags on the training set.
3. Enumerate candidate transformations of the form
   *"change tag X to tag Y when context condition C holds"*.
4. Greedily apply the transformation that removes the most errors (net error reduction),
   then repeat until the rule budget is exhausted.

The result is a compact cascade of contextual rules, e.g. *"change `NN` to `VBG` if the previous
tag is `VBZ`"*. Brill learning is slower than counting n-grams but yields interpretable rules and
usually beats a pure lexicon baseline by several points.
"""
    )
)

cells.append(
    code(
        r"""
start = time.perf_counter()
brill_tagger = BrillRuleTagger(max_rules=200, min_score=2).fit(split.train)
brill_train_seconds = time.perf_counter() - start
print(f"Brill training time: {brill_train_seconds:.2f} s")
print(f"Number of learned rules: {len(brill_tagger.learned_rules())}")

learned_rules = brill_tagger.learned_rules()
display(
    pd.DataFrame(
        {"#": range(1, 16), "Learned transformation rule": learned_rules[:15]}
    ).set_index("#")
)

brill_result = evaluate_tagger(brill_tagger.name, brill_tagger.tag, split.test, oov)
display(pd.DataFrame([brill_result.metrics()]).round(4))
print(classification_report(brill_result.gold, brill_result.predictions, zero_division=0))
"""
    )
)

cells.append(
    md(
        r"""
## 5. Approach 2: Statistical Tagging

### 5.1 Bigram Hidden Markov Model

An HMM is a generative sequence model. It assumes that the observed words are generated by a
hidden sequence of tags:

$$P(w_1,\dots,w_n,t_1,\dots,t_n) = \prod_{i=1}^{n} P(t_i \mid t_{i-1}) \, P(w_i \mid t_i)$$

with a special start state `<s>`. Tagging is the **decoding** problem

$$\hat{t}_{1:n} = \arg\max_{t_{1:n}} P(w_{1:n}, t_{1:n}),$$

solved exactly in $O(nT^2)$ with the Viterbi dynamic program:

$$\delta_i(t) = \max_{t'} \big[\delta_{i-1}(t') + \log P(t \mid t')\big] + \log P(w_i \mid t).$$

**Smoothing.** Raw maximum-likelihood estimates assign zero probability to unseen events. We use
add-$\alpha$ smoothing for start, transition, and emission distributions. Rare training words
(frequency < 2) are mapped to `<UNK>` so the model learns a probability mass for unknown words.

**OOV handling.** For a word never seen in training, the emission distribution is interpolated
between the learned `<UNK>` distribution and a suffix distribution conditioned on the final three
characters. This lets morphology inform the tag of an unseen word (e.g., `-ing` suggests `VBG`)
while still using contextual transitions. The interpolation weight is 0.35.

**Limitations.** The first-order Markov assumption is strong, emissions are conditioned only on the
current tag, and only the final three characters are used for OOV generalisation.
"""
    )
)

cells.append(
    code(
        r"""
start = time.perf_counter()
hmm_tagger = HMMTagger(alpha=0.1, min_word_freq=2, suffix_weight=0.35).fit(split.train)
hmm_train_seconds = time.perf_counter() - start
print(f"HMM training time: {hmm_train_seconds:.3f} s")
print(f"Vocabulary kept: {len(hmm_tagger.word_vocab):,} word types")
print(f"Tag state space: {len(hmm_tagger.tags)} tags")

display(pd.DataFrame(hmm_tagger.top_transitions(12)))
display(pd.DataFrame(hmm_tagger.top_emissions(10)))
"""
    )
)

cells.append(
    code(
        r"""
demo_sentence = [word for word, _ in split.test[0]]
demo_gold = [tag for _, tag in split.test[0]]
demo_predicted = hmm_tagger.tag(demo_sentence)
display(
    pd.DataFrame(
        {"Token": demo_sentence, "Gold": demo_gold, "HMM prediction": demo_predicted}
    )
)
"""
    )
)

cells.append(
    code(
        r"""
hmm_result = evaluate_tagger(hmm_tagger.name, hmm_tagger.tag, split.test, oov)
display(pd.DataFrame([hmm_result.metrics()]).round(4))
print(classification_report(hmm_result.gold, hmm_result.predictions, zero_division=0))
"""
    )
)

cells.append(
    md(
        r"""
### 5.2 N-gram tagger with back-off

A complementary statistical baseline: a trigram tagger backs off to a bigram tagger, then to a
unigram tagger, and finally to a default tag. Back-off avoids zero probabilities without an
explicit smoothing formula, but it cannot generalise to unseen word forms unless a suffix-based
component is added. It therefore illustrates the weakness of purely count-based models on OOV
tokens.
"""
    )
)

cells.append(
    code(
        r"""
start = time.perf_counter()
ngram_tagger = NgramBackoffTagger().fit(split.train)
ngram_train_seconds = time.perf_counter() - start
print(f"N-gram training time: {ngram_train_seconds:.3f} s")

ngram_result = evaluate_tagger(ngram_tagger.name, ngram_tagger.tag, split.test, oov)
display(pd.DataFrame([ngram_result.metrics()]).round(4))
print(classification_report(ngram_result.gold, ngram_result.predictions, zero_division=0))
"""
    )
)

cells.append(
    md(
        r"""
## 6. Approach 3: Neural Tagging with a Character-Aware BiLSTM

The neural tagger replaces sparse count tables with dense distributed representations and learns
the feature extractor end-to-end.

**Architecture**

1. **Word embeddings** (100 dimensions) map each token to a dense vector. Embeddings are learned
   from scratch on the training split, so the experiment does not depend on downloading external
   vectors.
2. **Character BiLSTM** (2 x 50 hidden units) reads the characters of each word and max-pools the
   hidden states. This gives every word a morphological representation and, crucially, allows the
   model to tag unseen words from their spelling.
3. The word embedding and character representation are concatenated and passed through dropout.
4. A **word-level BiLSTM** (2 x 128 hidden units) contextualises each token using both left and
   right context.
5. A linear layer produces one score per tag; softmax gives $P(t_i \mid w_{1:n})$.
6. Training minimises token-level cross-entropy with padded positions masked out, using Adam,
   gradient clipping, and early stopping on validation accuracy.

**Why it should handle OOV better:** an unseen word is not an unknown symbol. Its characters are
composed into a representation, so suffixes and word shapes provide evidence for the tag. Context
from the bidirectional encoder resolves the remaining ambiguity.

**Limitations:** the model is a black box, requires more compute, has many hyperparameters, and its
softmax outputs are not constrained to valid tag sequences (no CRF layer).
"""
    )
)

cells.append(
    code(
        r"""
start = time.perf_counter()
neural_model, vocabs, history = train_bilstm(
    split.train,
    split.val,
    epochs=12,
    batch_size=32,
    learning_rate=1e-3,
    patience=4,
    seed=SEED,
    device="cpu",
)
neural_train_seconds = time.perf_counter() - start
print(f"Neural training time (including early stopping): {neural_train_seconds:.1f} s")
print(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
print(f"Words in neural vocabulary: {vocabs.n_words:,}")
print(f"Characters in neural vocabulary: {vocabs.n_chars}")
"""
    )
)

cells.append(
    code(
        r"""
history_frame = pd.DataFrame(
    {
        "Epoch": history["epoch"],
        "Training loss": history["train_loss"],
        "Validation accuracy": history["val_accuracy"],
    }
).set_index("Epoch")
display(history_frame.round(4))

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(history_frame.index, history_frame["Training loss"], marker="o", color=COLORS[3], label="Training loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Token-level cross-entropy loss", color=COLORS[3])
ax1.tick_params(axis="y", labelcolor=COLORS[3])

ax2 = ax1.twinx()
ax2.plot(history_frame.index, history_frame["Validation accuracy"], marker="s", color=COLORS[0], label="Validation accuracy")
ax2.set_ylabel("Validation accuracy", color=COLORS[0])
ax2.tick_params(axis="y", labelcolor=COLORS[0])
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax1.set_title("BiLSTM training dynamics")

lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_training_curve.png")
plt.show()
"""
    )
)

cells.append(
    code(
        r"""
torch.save(
    {
        "state_dict": neural_model.state_dict(),
        "word2idx": vocabs.word2idx,
        "char2idx": vocabs.char2idx,
        "tag2idx": vocabs.tag2idx,
    },
    ROOT / "assets" / "bilstm_pos.pt",
)

neural_tagger = NeuralTagger(neural_model, vocabs, device="cpu")
neural_result = evaluate_tagger(neural_tagger.name, neural_tagger.tag, split.test, oov)
display(pd.DataFrame([neural_result.metrics()]).round(4))
print(classification_report(neural_result.gold, neural_result.predictions, zero_division=0))
"""
    )
)

cells.append(
    code(
        r"""
example_sentences = [
    sentence for sentence in split.test if any(word in oov for word, _ in sentence)
][:3]

rows = []
for sentence in example_sentences:
    example_tokens = [word for word, _ in sentence]
    example_gold = [tag for _, tag in sentence]
    rows.append(
        {
            "Sentence": " ".join(example_tokens),
            "Gold tags": " ".join(example_gold),
            "Regex": " ".join(regex_tagger.tag(example_tokens)),
            "HMM": " ".join(hmm_tagger.tag(example_tokens)),
            "BiLSTM": " ".join(neural_tagger.tag(example_tokens)),
        }
    )

display(pd.DataFrame(rows).T)
"""
    )
)

cells.append(
    md(
        r"""
## 7. Comparative Analysis

All five taggers are evaluated on the same test sentences. The table below is the primary
empirical result of the project; the figures that follow separate overall accuracy, OOV accuracy,
per-tag behaviour, and throughput.
"""
    )
)

cells.append(
    code(
        r"""
results = [regex_result, brill_result, hmm_result, ngram_result, neural_result]
train_seconds = {
    regex_result.name: regex_train_seconds,
    brill_result.name: brill_train_seconds,
    hmm_result.name: hmm_train_seconds,
    ngram_result.name: ngram_train_seconds,
    neural_result.name: neural_train_seconds,
}

comparison = results_frame(results)
comparison["Training Seconds"] = comparison["Model"].map(train_seconds).round(2)
comparison["Error Rate"] = (1.0 - comparison["Accuracy"]).round(4)
comparison = comparison.sort_values("Accuracy", ascending=False).reset_index(drop=True)
display(comparison)

comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
print("Saved:", REPORT_DIR / "model_comparison.csv")
"""
    )
)

cells.append(
    code(
        r"""
metric_columns = ["Accuracy", "Macro F1", "OOV Accuracy"]
plot_frame = comparison.set_index("Model")[metric_columns]

fig, ax = plt.subplots(figsize=(12, 5))
plot_frame.plot.bar(ax=ax, color=COLORS[:3], width=0.8)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Overall, macro-averaged, and OOV accuracy by model")
ax.tick_params(axis="x", rotation=18)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_model_accuracy.png")
plt.show()
"""
    )
)

cells.append(
    code(
        r"""
speed_frame = comparison.set_index("Model")["Tokens/Second"].sort_values()

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.barh(speed_frame.index, speed_frame.values, color=COLORS[4])
ax.set_xlabel("Test tokens tagged per second (CPU, one sentence at a time)")
ax.set_title("Inference throughput")
for bar, value in zip(bars, speed_frame.values):
    ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,.0f}", va="center", fontsize=9)
ax.set_xlim(0, speed_frame.max() * 1.18)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_speed.png")
plt.show()

training_frame = comparison.set_index("Model")["Training Seconds"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.barh(training_frame.index, training_frame.values, color=COLORS[1])
ax.set_xlabel("Training time (seconds, CPU)")
ax.set_title("Training cost")
for bar, value in zip(bars, training_frame.values):
    ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f} s", va="center", fontsize=9)
ax.set_xlim(0, max(training_frame.max() * 1.18, 1))
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_training_time.png")
plt.show()
"""
    )
)

cells.append(
    code(
        r"""
tag_scores = per_tag_frame(results)
display(tag_scores.head(20))

model_names = [result.name for result in results]
top_tags = tag_scores.head(18)
heatmap_values = top_tags[model_names].to_numpy()

fig, ax = plt.subplots(figsize=(11, 7))
image = ax.imshow(heatmap_values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels([name.replace(" (", "\n(") for name in model_names], fontsize=8)
ax.set_yticks(range(len(top_tags)))
ax.set_yticklabels(top_tags.index, fontsize=8)
ax.set_title("Per-tag F1 for the 18 most frequent test tags")
ax.grid(False)

for row in range(heatmap_values.shape[0]):
    for column in range(heatmap_values.shape[1]):
        value = heatmap_values[row, column]
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value < 0.55 else "black")

fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="F1 score")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_per_tag_f1.png")
plt.show()
"""
    )
)

cells.append(
    code(
        r"""
def plot_confusion(result, ax, top_n=15):
    tag_order = [
        tag
        for tag, _ in sorted(result.per_tag_support.items(), key=lambda item: -item[1])
        if tag in result.labels
    ][:top_n]
    indices = [result.labels.index(tag) for tag in tag_order]
    matrix = np.asarray(result.confusion)[np.ix_(indices, indices)].astype(float)
    matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tag_order)))
    ax.set_xticklabels(tag_order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tag_order)))
    ax.set_yticklabels(tag_order, fontsize=8)
    ax.set_xlabel("Predicted tag")
    ax.set_ylabel("Gold tag")
    ax.set_title(f"{result.name}: row-normalised confusion")
    ax.grid(False)
    return image


fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
image = plot_confusion(hmm_result, axes[0])
plot_confusion(neural_result, axes[1])
fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Proportion of gold tag")
fig.savefig(FIG_DIR / "fig_confusion_matrices.png")
plt.show()
"""
    )
)

cells.append(
    code(
        r"""
from IPython.display import display as ipy_display

for result in (brill_result, hmm_result, neural_result):
    print("=" * 80)
    print(result.name, "- first errors on the test set")
    ipy_display(pd.DataFrame(error_examples(result, limit=10)))

oov_rows = []
for token, gold, hmm_prediction, neural_prediction in zip(
    test_tokens, hmm_result.gold, hmm_result.predictions, neural_result.predictions
):
    if token in oov:
        oov_rows.append(
            {
                "OOV token": token,
                "Gold": gold,
                "HMM": hmm_prediction,
                "BiLSTM": neural_prediction,
            }
        )
        if len(oov_rows) >= 20:
            break

oov_error_frame = pd.DataFrame(oov_rows)
display(oov_error_frame)
"""
    )
)

cells.append(
    code(
        r"""
from scipy.stats import binomtest

def mcnemar_test(result_a, result_b):
    gold = np.asarray(result_a.gold)
    correct_a = gold == np.asarray(result_a.predictions)
    correct_b = gold == np.asarray(result_b.predictions)
    solved_by_b = int(np.sum(~correct_a & correct_b))
    solved_by_a = int(np.sum(correct_a & ~correct_b))
    discordant = solved_by_a + solved_by_b
    p_value = binomtest(solved_by_b, discordant, 0.5).pvalue if discordant else 1.0
    return {
        "Comparison": f"{result_a.name} vs {result_b.name}",
        "Only first model correct": solved_by_a,
        "Only second model correct": solved_by_b,
        "Discordant tokens": discordant,
        "McNemar exact p-value": p_value,
    }

significance = pd.DataFrame(
    [
        mcnemar_test(hmm_result, neural_result),
        mcnemar_test(brill_result, hmm_result),
        mcnemar_test(brill_result, neural_result),
    ]
)
significance["Significant at 0.05"] = significance["McNemar exact p-value"] < 0.05
display(significance.style.format({"McNemar exact p-value": "{:.3e}"}))
"""
    )
)

cells.append(
    code(
        r"""
best = comparison.iloc[0]
regex_row = comparison[comparison["Model"] == regex_result.name].iloc[0]
brill_row = comparison[comparison["Model"] == brill_result.name].iloc[0]
hmm_row = comparison[comparison["Model"] == hmm_result.name].iloc[0]
neural_row = comparison[comparison["Model"] == neural_result.name].iloc[0]

summary_lines = [
    "### Empirical takeaways",
    "",
    f"- **Best overall accuracy:** {best['Model']} with {best['Accuracy']:.2%} "
    f"(error rate {best['Error Rate']:.2%}).",
    f"- **Hand-written rules:** {regex_row['Accuracy']:.2%} accuracy; the symbolic approach is "
    f"instantaneous and fully inspectable but brittle.",
    f"- **Brill TBL:** {brill_row['Accuracy']:.2%} accuracy with {len(learned_rules)} learned rules; "
    f"rule learning closes about "
    f"{(brill_row['Accuracy'] - regex_row['Accuracy']) * 100:.1f} percentage points over the "
    f"hand-written cascade.",
    f"- **HMM:** {hmm_row['Accuracy']:.2%} accuracy at {hmm_row['Tokens/Second']:,.0f} tokens/s, "
    f"with OOV accuracy {hmm_row['OOV Accuracy']:.2%} thanks to suffix-smoothed emissions.",
    f"- **BiLSTM:** {neural_row['Accuracy']:.2%} accuracy and {neural_row['OOV Accuracy']:.2%} on "
    f"OOV tokens, at {neural_row['Tokens/Second']:,.0f} tokens/s and "
    f"{neural_row['Training Seconds']:.1f} s of training.",
    f"- **Accuracy vs. cost:** the neural model reduces the HMM error rate by "
    f"{(hmm_row['Error Rate'] - neural_row['Error Rate']) / hmm_row['Error Rate']:.1%} relative, "
    f"but its training cost is roughly "
    f"{neural_row['Training Seconds'] / max(hmm_row['Training Seconds'], 1e-9):,.0f}x larger "
    f"and its inference is "
    f"{hmm_row['Tokens/Second'] / max(neural_row['Tokens/Second'], 1e-9):,.0f}x slower on CPU.",
    f"- **Interpretability ranking:** hand-written rules > Brill rules > HMM transition/emission "
    f"tables > BiLSTM weights.",
]

from IPython.display import Markdown, display as display_markdown
display_markdown(Markdown("\n".join(summary_lines)))
"""
    )
)

cells.append(
    md(
        r"""
## 8. Discussion

### 8.1 Rules versus statistics versus neural networks

The empirical ordering matches the theoretical expectation, but the size of each step is
informative:

- **Hand-written rules are a strong zero-cost baseline but plateau quickly.** The regex cascade
  encodes useful morphology, yet it cannot resolve contextual ambiguity, so it produces many
  `NN` defaults. This is the "knowledge acquisition bottleneck" in symbolic NLP.
- **Learned symbolic rules are much stronger than hand-written ones.** Brill TBL recovers most of
  the gap without giving up interpretability, because its transformations are induced from the
  actual error distribution of the corpus rather than from a linguist's intuition.
- **A well-smoothed HMM is a surprisingly tough baseline.** Transition structure (for example,
  determiners precede nouns/adjectives) plus suffix-aware unknown-word emissions is enough to
  approach the neural model's overall accuracy. The gap is concentrated in hard ambiguities and
  OOV tokens.
- **The BiLSTM wins on overall and OOV accuracy.** Character composition gives it a genuine
  advantage on unseen words, and bidirectional context resolves local ambiguities that a first-order
  HMM cannot represent. The price is training time, inference cost, and opacity.

### 8.2 Error patterns and confusion matrices

The confusion matrices expose systematic confusions rather than random mistakes. Typical
patterns on the PTB tagset include:

- **`NN` vs `NNP`:** whether a capitalised token is a common or proper noun depends on the
  sentence and the corpus annotation convention. Hand-written capitalisation rules over-predict
  `NNP`; the neural model can use context to correct many of these cases.
- **`VBD` vs `VBN` vs `VB`:** past tense, past participle, and base form are morphologically
  identical for many verbs (*walked*, *put*). Only syntactic context disambiguates them.
- **`JJ` vs `NN`:** nouns frequently modify other nouns (*trading company*), and adjectives can be
  nominalised.
- **OOV morphology:** an unseen `-ing` form is usually a verb, but can be a noun. The HMM's suffix
  table is a coarse approximation; the character BiLSTM learns non-linear character patterns.

The neural model's diagonal is visibly sharper for noun and verb categories, which is where most
OOV tokens occur. Errors that remain are mostly genuinely ambiguous tokens that would require
sentence-level semantics or annotation-convention knowledge.

### 8.3 Engineering trade-offs

| Dimension | Rule-based (regex) | Brill TBL | HMM | BiLSTM |
|---|---|---|---|---|
| Training cost | None | Seconds | Sub-second | Minutes (CPU) |
| Inference cost | Extremely low | Very low | Very low | Higher (tensor overhead) |
| OOV strategy | Hard-coded suffixes | Learned suffix/capitalisation rules | `<UNK>` + suffix emissions | Character composition |
| Interpretability | High | High (rule list) | Moderate (probability tables) | Low |
| Accuracy ceiling | Low | Medium-high | High | Highest |
| Data requirement | None | Small | Small-medium | Medium-large |
| Main failure mode | Contextual ambiguity | Annotation inconsistency | Markov assumption, sparse counts | Compute, opacity, overfitting |

### 8.4 Threats to validity

- The NLTK Treebank sample is small (about 3,900 sentences), so absolute numbers will differ from
  results on the full Penn Treebank or Universal Dependencies.
- A single fixed train/test split is used for reproducibility; a cross-validated study would give
  confidence intervals, but triples the neural training cost.
- The neural model uses embeddings learned from scratch rather than pretrained vectors, which
  deliberately avoids external downloads but leaves accuracy on the table.
- Timings are single-run measurements on one CPU; they reflect wall-clock behaviour in this
  environment rather than a controlled benchmark.
- The PTB tagset contains annotation-convention distinctions (for example, `NNP` for
  sentence-initial capitalised common nouns in some contexts) that no model can infer from the
  token alone.
"""
    )
)

cells.append(
    md(
        r"""
## 9. Conclusion and Future Work

This project delivered a controlled comparison of three POS tagging paradigms on the same corpus,
split, and evaluation protocol:

1. **Rule-based taggers** are transparent and cheap. Hand-written regex rules are a useful
   baseline, while Brill TBL shows that symbolic rules can be learned and remain readable.
2. **Statistical taggers** are excellent value: a bigram HMM with add-alpha smoothing and
   suffix-aware unknown-word emissions is fast to train, fast to run, and competitive in accuracy.
3. **Neural taggers** deliver the best accuracy and, most visibly, the best OOV accuracy, because
   character-level composition and bidirectional context handle unseen morphology and ambiguity.
   They are the most expensive to train and the least interpretable.

The practical recommendation is therefore **conditional**: use rules or an HMM when training data
and compute are scarce and transparency matters; use a character-aware neural tagger when accuracy
on noisy, open-vocabulary text is the priority.

### Future work

- Add a **CRF layer** or structured attention to enforce valid tag transitions and model tag
  dependencies jointly.
- Initialise with **pretrained embeddings** (GloVe, fastText) or fine-tune a **transformer**
  (DistilBERT) to measure how much transfer learning helps on this small corpus.
- Evaluate on **Universal Dependencies** and on domain-shifted text to test robustness.
- Report **bootstrap confidence intervals** and repeated splits instead of single-point estimates.
- Explore **distillation** from the neural tagger into a small HMM or rule cascade for cheap
  deployment.
- Profile **batched inference** for the neural model; sentence-by-sentence tagging understates its
  throughput advantage on GPUs.
"""
    )
)

cells.append(
    md(
        r"""
## 10. Reproducibility and Saved Artifacts

The notebook saves all numbers and figures needed for the report and the screenshot PDF:

| Artifact | Location |
|---|---|
| Comparison table | `reports/model_comparison.csv` |
| Full metrics (JSON) | `reports/results.json` |
| Trained BiLSTM weights | `assets/bilstm_pos.pt` |
| Figures | `assets/figures/*.png` |

To reproduce: install `requirements.txt`, launch Jupyter, and run all cells. The dataset is
downloaded automatically by NLTK on first use. Neural training uses a fixed seed, but CPU
non-determinism can cause small run-to-run differences in the last digit of accuracy.

### Deliverables checklist

- [x] Complete Jupyter notebook (this file).
- [ ] Project report PDF (`reports/POS_Tagging_Report.pdf`).
- [ ] Output screenshots PDF (`reports/output_screenshots.pdf`).
- [ ] Demo video MP4 (see `reports/demo_video_script.md`).
- [ ] GitHub repository link in the report and README.
"""
    )
)

cells.append(
    code(
        r"""
environment = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "platform": platform.platform(),
    "processor": platform.processor() or platform.machine(),
}

payload = {
    "dataset": split.summary(),
    "oov": {
        "word_types": len(oov),
        "tokens": int(test_oov_mask.sum()),
        "token_rate": float(test_oov_mask.mean()),
    },
    "environment": environment,
    "seed": SEED,
    "results": [],
    "neural_history": history,
    "mcnemar": significance.drop(columns=["Significant at 0.05"]).to_dict(orient="records"),
}

for result in results:
    record = result.metrics()
    record["Training Seconds"] = train_seconds[result.name]
    record["Per Tag F1"] = result.per_tag_f1
    payload["results"].append(record)

with open(REPORT_DIR / "results.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

(REPORT_DIR / "model_comparison.md").write_text(
    comparison.to_string(index=False), encoding="utf-8"
)

figures = sorted(path.name for path in FIG_DIR.glob("*.png"))
print("Saved results.json and model_comparison.md")
print("Figures:")
for figure in figures:
    print(" -", figure)
print()
print(comparison.to_string(index=False))
"""
    )
)

cells.append(
    md(
        r"""
## References

1. Brill, E. (1992). *A simple rule-based part of speech tagger.* Proceedings of the Third
   Conference on Applied Natural Language Processing, 152-155.
2. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft),
   chapters on sequence labelling and POS tagging.
3. Loper, E., & Bird, S. (2002). *NLTK: The Natural Language Toolkit.* Proceedings of ACL
   Workshop on Effective Tools and Methodologies for Teaching NLP.
4. Marcus, M. P., Santorini, B., & Marcinkiewicz, M. A. (1993). *Building a large annotated corpus
   of English: The Penn Treebank.* Computational Linguistics, 19(2), 313-330.
5. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). *Neural
   architectures for named entity recognition.* NAACL-HLT.
6. Plank, B., Sogaard, A., & Goldberg, Y. (2016). *Multilingual part-of-speech tagging with
   bidirectional long short-term memory models and auxiliary loss.* ACL.
7. Huang, Z., Xu, W., & Yu, K. (2015). *Bidirectional LSTM-CRF models for sequence tagging.*
   arXiv:1508.01991.
"""
    )
)

notebook = new_notebook(cells=cells)
notebook.metadata.update(
    {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "title": "Rule-Based vs Statistical vs Neural POS Tagging",
    }
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells")

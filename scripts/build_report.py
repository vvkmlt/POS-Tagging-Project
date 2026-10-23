"""Build the formatted project report HTML from the executed notebook results."""

from __future__ import annotations

import base64
import html
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "assets" / "figures"
RESULTS = json.loads((REPORT_DIR / "results.json").read_text(encoding="utf-8"))

results = sorted(RESULTS["results"], key=lambda row: -row["Accuracy"])
for row in results:
    row["Error Rate"] = 1.0 - row["Accuracy"]
by_model = {row["Model"]: row for row in results}
dataset = RESULTS["dataset"]
oov = RESULTS["oov"]
environment = RESULTS["environment"]
significance = RESULTS["mcnemar"]

regex = by_model["Rule-Based (Regex)"]
brill = by_model["Rule-Based (Brill TBL)"]
hmm = by_model["Statistical (HMM)"]
ngram = by_model["Statistical (N-gram Backoff)"]
neural = by_model["Neural (BiLSTM)"]

rule_gain = (brill["Accuracy"] - regex["Accuracy"]) * 100
neural_gain = (neural["Accuracy"] - hmm["Accuracy"]) * 100
error_reduction = (hmm["Error Rate"] - neural["Error Rate"]) / hmm["Error Rate"] * 100
oov_gain = (neural["OOV Accuracy"] - hmm["OOV Accuracy"]) * 100
speed_ratio = hmm["Tokens/Second"] / neural["Tokens/Second"]
train_ratio = neural["Training Seconds"] / max(hmm["Training Seconds"], 1e-9)

CSS = """
@page { size: A4; margin: 20mm 17mm 18mm 17mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.55; color: #1c1c1c; margin: 0;
}
h1, h2, h3, h4 { font-family: "Segoe UI", Arial, sans-serif; color: #16324f; line-height: 1.25; }
h1 { font-size: 22pt; margin: 0 0 8pt; }
h2 { font-size: 15pt; margin: 22pt 0 8pt; padding-bottom: 4pt; border-bottom: 1.5pt solid #16324f; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 16pt 0 6pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p { margin: 0 0 8pt; text-align: justify; }
ul, ol { margin: 4pt 0 9pt 18pt; padding: 0; }
li { margin-bottom: 3pt; }
.cover { height: 245mm; display: flex; flex-direction: column; justify-content: space-between; page-break-after: always; text-align: center; }
.cover .top { margin-top: 18mm; }
.cover .institution { font-family: "Segoe UI", Arial, sans-serif; font-size: 13pt; letter-spacing: 0.5pt; text-transform: uppercase; color: #16324f; font-weight: 600; }
.cover .department { font-family: "Segoe UI", Arial, sans-serif; font-size: 11pt; color: #555; margin-top: 4pt; }
.cover .rule { width: 45mm; height: 2pt; background: #b07d3c; margin: 14pt auto; }
.cover .kicker { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; letter-spacing: 2pt; text-transform: uppercase; color: #b07d3c; font-weight: 600; }
.cover h1 { font-size: 25pt; max-width: 150mm; margin: 12pt auto 10pt; }
.cover .subtitle { font-size: 12pt; color: #444; font-style: italic; max-width: 140mm; margin: 0 auto; }
.cover .meta { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; display: inline-block; text-align: left; margin: 0 auto; }
.cover .meta td { padding: 3pt 10pt 3pt 0; vertical-align: top; }
.cover .meta td:first-child { color: #666; }
.cover .foot { font-family: "Segoe UI", Arial, sans-serif; font-size: 9pt; color: #666; margin-bottom: 6mm; }
.abstract { page-break-after: always; }
.abstract p { font-size: 10.5pt; }
.keywords { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.5pt; color: #333; background: #f3f6f9; border-left: 3pt solid #16324f; padding: 8pt 10pt; margin-top: 12pt; }
table.data { width: 100%; border-collapse: collapse; margin: 8pt 0 12pt; font-family: "Segoe UI", Arial, sans-serif; font-size: 8.6pt; page-break-inside: avoid; }
table.data th { background: #16324f; color: #fff; text-align: left; padding: 5pt 5pt; font-weight: 600; }
table.data td { padding: 4pt 5pt; border-bottom: 0.5pt solid #d5dbe1; vertical-align: top; }
table.data tr:nth-child(even) td { background: #f6f8fa; }
table.data td.num, table.data th.num { text-align: right; }
table.data caption { caption-side: top; text-align: left; font-family: "Segoe UI", Arial, sans-serif; font-size: 8.8pt; font-weight: 600; color: #16324f; padding-bottom: 4pt; }
.callout { background: #f4f7f4; border-left: 3pt solid #3d6b45; padding: 8pt 11pt; margin: 10pt 0 12pt; page-break-inside: avoid; }
.callout strong { color: #2d5233; }
.note { background: #fbf7ef; border-left: 3pt solid #b07d3c; padding: 8pt 11pt; margin: 10pt 0 12pt; page-break-inside: avoid; font-size: 9.8pt; }
figure { margin: 10pt 0 14pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 0.5pt solid #d5dbe1; }
figcaption { font-family: "Segoe UI", Arial, sans-serif; font-size: 8.6pt; color: #555; margin-top: 4pt; text-align: left; }
.formula { font-family: "Cambria Math", "Times New Roman", serif; font-size: 10.5pt; text-align: center; margin: 6pt 0 9pt; font-style: italic; }
.small { font-size: 9pt; color: #555; }
.refs { font-size: 9.5pt; }
.refs li { margin-bottom: 6pt; }
.page-break { page-break-before: always; }
code { font-family: "Consolas", monospace; font-size: 9pt; background: #f2f4f6; padding: 0 2pt; }
pre { font-family: "Consolas", monospace; font-size: 8.5pt; background: #f5f7f9; border: 0.5pt solid #dde3e8; padding: 7pt; white-space: pre-wrap; page-break-inside: avoid; }
"""


def escape(value: object) -> str:
    return html.escape(str(value))


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def image(filename: str, caption: str, width: str = "100%") -> str:
    path = FIG_DIR / filename
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" style="width:{width}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def table(headers, rows, caption: str, numeric_columns=()) -> str:
    head = "".join(
        f'<th class="num">{escape(cell)}</th>' if index in numeric_columns else f"<th>{escape(cell)}</th>"
        for index, cell in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{escape(cell)}</td>' if index in numeric_columns else f"<td>{escape(cell)}</td>"
            for index, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f"<table class='data'><caption>{escape(caption)}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


dataset_rows = [
    ["Corpus", "NLTK Penn Treebank sample (PTB tagset)"],
    ["Sentences (total / train / validation / test)", f"{dataset['train_sentences'] + dataset['val_sentences'] + dataset['test_sentences']:,} / {dataset['train_sentences']:,} / {dataset['val_sentences']:,} / {dataset['test_sentences']:,}"],
    ["Tokens (train / validation / test)", f"{dataset['train_tokens']:,} / {dataset['val_tokens']:,} / {dataset['test_tokens']:,}"],
    ["Split protocol", "Sentence-level 80/10/10, fixed seed 42"],
    ["Train tagset size", "45 PTB tags"],
    ["Test OOV tokens", f"{oov['tokens']:,} ({oov['token_rate'] * 100:.2f}% of test tokens)"],
    ["Test OOV word types", f"{oov['word_types']:,}"],
]

hyperparameter_rows = [
    ["Rule-based (regex)", "No parameters", "Ordered lexicon + suffix cascade; default tag NN", "0 s"],
    ["Brill TBL", "max_rules=200, min_score=2", "Unigram baseline (back-off NN), nltkdemo18 templates, deterministic", f"{brill['Training Seconds']:.2f} s"],
    ["HMM", "alpha=0.1, min_word_freq=2, suffix_weight=0.35", "Bigram transitions; add-alpha smoothing; Viterbi decoding", f"{hmm['Training Seconds']:.3f} s"],
    ["N-gram back-off", "cutoff=1 for trigram/bigram/unigram", "Trigram -> bigram -> unigram -> NN", f"{ngram['Training Seconds']:.2f} s"],
    ["BiLSTM", "word_dim=100, char_dim=30, char_hidden=50, hidden=128, dropout=0.33, batch=32, Adam lr=1e-3, max 12 epochs, patience=4", "Character BiLSTM + word BiLSTM; cross-entropy with masked padding; gradient clipping 5.0", f"{neural['Training Seconds']:.1f} s"],
]

main_rows = [
    [
        row["Model"],
        pct(row["Accuracy"]),
        pct(row["Macro F1"]),
        pct(row["Weighted F1"]),
        pct(row["OOV Accuracy"]),
        number(row["Training Seconds"]),
        number(row["Tokens/Second"], 0),
        pct(row["Error Rate"]),
    ]
    for row in results
]

significance_rows = [
    [
        row["Comparison"],
        row["Only first model correct"],
        row["Only second model correct"],
        row["Discordant tokens"],
        f"{row['McNemar exact p-value']:.2e}",
        "Yes" if row["McNemar exact p-value"] < 0.05 else "No",
    ]
    for row in significance
]

qualitative_rows = [
    ["Training cost", "None", "Seconds", "Sub-second", "Minutes (CPU)"],
    ["Inference cost", "Very low", "Very low", "Very low", "Moderate"],
    ["OOV mechanism", "Hand-written suffix rules", "Learned contextual rules", "UNK + suffix emissions", "Character composition"],
    ["Interpretability", "High", "High (rule list)", "Moderate (tables)", "Low"],
    ["Accuracy ceiling", "Low", "Medium-high", "High", "Highest"],
    ["Main failure mode", "Contextual ambiguity", "No morphological generalisation", "Markov and lexical sparsity", "Compute, opacity"],
]

report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A Theoretical and Empirical Comparison of Rule-Based, Statistical, and Neural POS Tagging</title>
<style>{CSS}</style>
</head>
<body>

<section class="cover">
  <div class="top">
    <div class="institution">EASA COLLEGE OF ENGINEERING AND TECHNOLOGY</div>
    <div class="department">Department of Computer Science and Engineering (AI &amp; ML)</div>
    <div class="rule"></div>
    <div class="kicker">Term Project Report</div>
    <h1>A Theoretical and Empirical Comparison of Rule-Based, Statistical, and Neural Part-of-Speech Tagging</h1>
    <div class="subtitle">An implementation study on the Penn Treebank with a regex tagger, Brill transformation-based learning, a smoothed HMM, N-gram back-off, and a character-aware BiLSTM</div>
  </div>

  <table class="meta">
    <tr><td>Submitted by</td><td>VENKATESH V</td></tr>
    <tr><td>Roll number</td><td>E720524AM034</td></tr>
    <tr><td>Course</td><td>B.E - CSE (AI &amp; ML) &mdash; U24DSP13</td></tr>
    <tr><td>Date</td><td>{date.today().strftime('%d %B %Y')}</td></tr>
    <tr><td>Repository</td><td><a href="https://github.com/vvkmlt/POS-Tagging-Project">github.com/vvkmlt/POS-Tagging-Project</a></td></tr>
  </table>

  <div class="foot">Submitted in partial fulfilment of the requirements of the term project.</div>
</section>

<section class="abstract">
  <h2 style="margin-top:0">Abstract</h2>
  <p>Part-of-speech (POS) tagging is a core sequence-labelling task in natural language processing. This report presents a controlled theoretical and empirical comparison of three generations of tagging approaches: rule-based methods, statistical sequence models, and neural sequence models. Five taggers were implemented on the same 3,914-sentence Penn Treebank sample and evaluated on an identical held-out test split: a hand-written regex and lexicon tagger, a Brill transformation-based learner, a bigram Hidden Markov Model with add-alpha smoothing and suffix-aware unknown-word emissions, a trigram tagger with back-off, and a character-aware bidirectional LSTM.</p>
  <p>The empirical ordering follows the theoretical expectation: the hand-written rule cascade reached {pct(regex['Accuracy'])} accuracy, Brill TBL reached {pct(brill['Accuracy'])}, the smoothed HMM reached {pct(hmm['Accuracy'])}, and the BiLSTM reached {pct(neural['Accuracy'])}. The neural model reduced the HMM error rate by {error_reduction:.1f}% relative and improved out-of-vocabulary (OOV) accuracy from {pct(hmm['OOV Accuracy'])} to {pct(neural['OOV Accuracy'])}. However, the result also exposes important nuances. The regex tagger generalised to unseen words better than Brill TBL ({pct(regex['OOV Accuracy'])} vs {pct(brill['OOV Accuracy'])}) because its suffix rules apply to any token, whereas Brill's learned contextual rules depend on lexical and contextual patterns observed in training. The study quantifies the trade-offs between accuracy, OOV robustness, training cost, inference throughput, and interpretability, and shows that a well-smoothed HMM remains a strong low-cost baseline. All code, data splits, figures, and results are reproducible from the accompanying notebook.</p>
  <div class="keywords"><strong>Keywords:</strong> part-of-speech tagging; rule-based NLP; Brill transformation-based learning; Hidden Markov Model; Viterbi decoding; BiLSTM; character embeddings; out-of-vocabulary words; empirical evaluation.</div>
</section>

<h2>1. Introduction</h2>
<h3>1.1 Problem statement</h3>
<p>POS tagging assigns a grammatical category to every token in a sentence. The task is deceptively simple: most tokens are frequent and locally predictable, yet a long tail of ambiguous words requires syntactic and semantic context. The same surface form can be a noun or a verb (<em>book</em>, <em>run</em>), a past tense or a past participle (<em>walked</em>), or a proper or common noun depending on annotation conventions.</p>
<p>Three paradigms offer different answers to this ambiguity. Rule-based systems encode linguistic knowledge as deterministic patterns. Statistical systems estimate probability distributions over tags and words from annotated data. Neural systems learn dense representations and contextual encoders end-to-end. Each paradigm makes different assumptions about data, compute, and interpretability, which makes POS tagging an ideal case study for comparing them.</p>

<h3>1.2 Significance</h3>
<p>POS tags are consumed by parsers, named-entity recognisers, machine translation systems, and information-extraction pipelines. Tagging errors propagate downstream, and the choice of tagger is often constrained by deployment conditions: a mobile application may need millisecond inference and a tiny memory footprint, while an offline corpus annotation pipeline may prioritise accuracy. A quantitative comparison on a common benchmark therefore has both pedagogical and engineering value.</p>

<h3>1.3 Objectives</h3>
<ol>
  <li>Implement representative rule-based, statistical, and neural POS taggers on a standard corpus.</li>
  <li>Compare them under an identical train/test protocol using accuracy, macro F1, OOV accuracy, training time, and inference throughput.</li>
  <li>Analyse error patterns and confusion matrices to explain <em>why</em> the differences arise, rather than only reporting aggregate scores.</li>
  <li>Derive practical recommendations about accuracy, cost, and interpretability trade-offs.</li>
</ol>

<h3>1.4 Contributions</h3>
<ul>
  <li>A reproducible notebook and modular source package implementing five taggers with no external pretrained resources.</li>
  <li>A suffix-smoothed HMM that substantially improves OOV tagging over a standard N-gram back-off tagger.</li>
  <li>An empirical demonstration that character-level neural modelling dominates on OOV words, while a smoothed HMM remains competitive at a fraction of the cost.</li>
  <li>A quantified interpretability-versus-accuracy analysis, including a statistical significance test on paired predictions.</li>
</ul>

<h2>2. Background and Related Work</h2>
<h3>2.1 Rule-based tagging</h3>
<p>Early taggers used hand-crafted disambiguation rules and lexicons. A rule-based tagger is transparent and requires no annotated training data, but it suffers from the knowledge-acquisition bottleneck: writing rules for every ambiguity is expensive, and rules often interact unpredictably. Brill's transformation-based learning (TBL) addressed this by starting from a simple baseline and greedily learning contextual transformations that reduce training error (Brill, 1992). The learned rules remain human-readable, making TBL a bridge between symbolic and statistical NLP.</p>

<h3>2.2 Statistical tagging</h3>
<p>Statistical taggers treat tagging as inference in a probabilistic model. Hidden Markov Models factor the joint probability of words and tags into transition and emission probabilities and decode with the Viterbi algorithm (Jurafsky and Martin, 2023). HMMs are fast and interpretable through their probability tables, but their first-order Markov assumption and sparse emission counts limit contextual generalisation. N-gram taggers with back-off provide a related count-based baseline, and maximum-entropy or conditional random field models relax the generative assumptions by modelling P(tags | words) discriminatively.</p>

<h3>2.3 Neural tagging</h3>
<p>Neural sequence models replace sparse count features with learned distributed representations. Bidirectional LSTMs encode left and right context (Huang et al., 2015), and character-level encoders compose morphological information for unseen words, which is particularly valuable for POS tagging (Plank et al., 2016; Lample et al., 2016). Transformer encoders and pretrained language models now define the state of the art, but they require substantially more data and compute. This project uses a character-aware BiLSTM because it trains on CPU in minutes, requires no external embeddings, and isolates the architectural benefit of character composition and bidirectional context.</p>

<h3>2.4 Positioning of this study</h3>
<p>Most textbook comparisons report only overall accuracy. This study holds the corpus, split, tagset, and evaluation API constant and adds OOV-specific accuracy, throughput, training cost, paired significance testing, and confusion analysis. It also includes both a hand-written and a learned rule-based tagger, which separates the effect of linguistic knowledge from the effect of rule induction.</p>

<h2>3. Methodology</h2>
<h3>3.1 Dataset and split</h3>
<p>The experiments use the Penn Treebank sample distributed with NLTK (Marcus et al., 1993). It contains {dataset['train_sentences'] + dataset['val_sentences'] + dataset['test_sentences']:,} sentences and {dataset['train_tokens'] + dataset['val_tokens'] + dataset['test_tokens']:,} tokens annotated with the 45-tag PTB tagset. Sentences were shuffled with a fixed seed and split 80/10/10 at the sentence level: {dataset['train_sentences']:,} training, {dataset['val_sentences']:,} validation, and {dataset['test_sentences']:,} test sentences. The validation split is used only for neural early stopping. The test split contains {oov['tokens']:,} OOV tokens ({oov['token_rate'] * 100:.2f}%), which makes OOV behaviour measurable.</p>

{table(['Property', 'Value'], dataset_rows, 'Table 1. Dataset and split statistics.')}

{image('fig_tag_distribution.png', 'Figure 1. Training-set tag distribution. The PTB tagset is highly imbalanced: nouns (NN, NNS, NNP) and punctuation dominate, while many tags have fewer than a thousand training examples. Macro F1 is therefore reported alongside accuracy.')}

<h3>3.2 Evaluation metrics</h3>
<p>For each model we report token-level accuracy, macro F1, weighted F1, OOV accuracy, training wall-clock time, and inference throughput. Macro F1 gives every tag equal weight and exposes weaknesses on rare categories; weighted F1 accounts for tag frequency. OOV accuracy is computed only on test tokens whose surface form never occurred in training. Throughput is measured by tagging one sentence at a time on CPU, which reflects the API used by downstream applications. Paired predictions are compared with an exact McNemar test to test whether accuracy differences are statistically significant.</p>

<h3>3.3 Implementations</h3>
<h4>Hand-written regex and lexicon tagger</h4>
<p>An ordered cascade applies punctuation and irregular-form lookups, closed-class word lists, and morphological regular expressions. For example, <code>-ing</code> maps to <code>VBG</code>, <code>-ed</code> to <code>VBD</code>, <code>-ly</code> to <code>RB</code>, capitalised forms to <code>NNP</code>, and otherwise <code>NN</code>. The tagger has no learned parameters and therefore zero training time.</p>

<h4>Brill transformation-based learning</h4>
<p>The baseline is a unigram tagger backed off to the default tag <code>NN</code>. Candidate transformations of the form <em>change tag X to Y when a contextual template fires</em> are evaluated on the training set, and the highest-scoring rule is applied greedily for a maximum of 200 rules. The resulting rule cascade is retained in human-readable form.</p>

<h4>Bigram Hidden Markov Model</h4>
<p>The model factorises the joint probability as</p>
<div class="formula">P(w<sub>1:n</sub>, t<sub>1:n</sub>) = &prod;<sub>i=1..n</sub> P(t<sub>i</sub> | t<sub>i-1</sub>) &middot; P(w<sub>i</sub> | t<sub>i</sub>)</div>
<p>with a start state. Transition, start, and emission distributions use add-alpha smoothing (&alpha; = 0.1). Training words with frequency below two are mapped to <code>&lt;UNK&gt;</code>. For OOV words, the emission probability is interpolated between the learned <code>&lt;UNK&gt;</code> distribution and a distribution conditioned on the final three characters, so that morphology can inform the tag while transition probabilities still enforce contextual consistency. Decoding uses Viterbi's dynamic program in log space.</p>

<h4>N-gram back-off</h4>
<p>A trigram tagger backs off to a bigram tagger, then a unigram tagger, then the default tag <code>NN</code>. This is a strong count-based baseline for seen contexts but has no explicit mechanism for unseen word forms beyond capitalisation patterns.</p>

<h4>Character-aware BiLSTM</h4>
<p>Each token is represented by a 100-dimensional word embedding concatenated with a character representation produced by a bidirectional LSTM over its characters (2 &times; 50 hidden units, max-pooled). The sequence of token representations is encoded by a word-level bidirectional LSTM (2 &times; 128 hidden units). Dropout (0.33) is applied before and after the word encoder, and a linear layer projects each contextual representation to tag scores. Training minimises token-level cross-entropy with padded positions ignored, using Adam (learning rate 10<sup>-3</sup>), gradient clipping at 5.0, a maximum of 12 epochs, and early stopping with patience 4 on validation accuracy. All embeddings are learned from scratch, so the experiment is self-contained and does not rely on network access to pretrained vectors.</p>

<h3>3.4 Reproducibility protocol</h3>
<p>The corpus, split seed, model seeds, and evaluation functions are fixed in the notebook. The dataset is downloaded automatically by NLTK. Results are written to JSON and CSV, and all figures are regenerated by executing the notebook from top to bottom. The neural checkpoint is saved to <code>assets/bilstm_pos.pt</code>.</p>

<h2>4. Experimental Setup</h2>
<table class="data">
<caption>Table 2. Implementation and hyperparameter configuration.</caption>
<thead><tr><th>Model</th><th>Key hyperparameters</th><th>Design choices</th><th class="num">Training time</th></tr></thead>
<tbody>
{''.join(f'<tr><td>{escape(row[0])}</td><td>{escape(row[1])}</td><td>{escape(row[2])}</td><td class="num">{escape(row[3])}</td></tr>' for row in hyperparameter_rows)}
</tbody>
</table>

<p class="small"><strong>Environment.</strong> Python {escape(environment['python'])}, PyTorch {escape(environment['torch'])}, NumPy {escape(environment['numpy'])}, pandas {escape(environment['pandas'])}. Platform: {escape(environment['platform'])}. All models were trained and evaluated on CPU with no GPU acceleration. Timings are single-run wall-clock measurements and should be interpreted comparatively rather than as absolute benchmarks.</p>

<h2>5. Results</h2>
<h3>5.1 Overall performance</h3>
{table(
    ['Model', 'Accuracy', 'Macro F1', 'Weighted F1', 'OOV accuracy', 'Train (s)', 'Tokens/s', 'Error rate'],
    main_rows,
    'Table 3. Test-set results. All models are evaluated on the same 9,995 test tokens.',
    numeric_columns=(1, 2, 3, 4, 5, 6, 7),
)}

{image('fig_model_accuracy.png', 'Figure 2. Overall accuracy, macro F1, and OOV accuracy by model. The BiLSTM leads on all three metrics; the hand-written regex tagger trails substantially on overall accuracy but is not the weakest model on OOV tokens.')}

<p>The BiLSTM achieved the best overall accuracy ({pct(neural['Accuracy'])}), followed by the smoothed HMM ({pct(hmm['Accuracy'])}), Brill TBL ({pct(brill['Accuracy'])}), N-gram back-off ({pct(ngram['Accuracy'])}), and the hand-written regex tagger ({pct(regex['Accuracy'])}). The gap between hand-written rules and learned Brill rules is {rule_gain:.1f} percentage points, which quantifies the value of inducing context-sensitive rules from annotated data. The neural model improves on the HMM by {neural_gain:.2f} percentage points, corresponding to a {error_reduction:.1f}% relative reduction in error.</p>
<p>Macro F1 reveals a different pattern from accuracy. Brill TBL reaches {pct(brill['Macro F1'])}, close to the BiLSTM's {pct(neural['Macro F1'])} and above the HMM's {pct(hmm['Macro F1'])}. Because Brill's transformations are driven by error reduction across the tagset, they can improve rare tags even when overall accuracy remains lower. The regex tagger's macro F1 ({pct(regex['Macro F1'])}) is much lower than its accuracy because it defaults many ambiguous words to <code>NN</code> and performs poorly on several mid-frequency categories.</p>

<h3>5.2 Out-of-vocabulary behaviour</h3>
<div class="callout">
<strong>Key finding.</strong> The BiLSTM tagged unseen words with {pct(neural['OOV Accuracy'])} accuracy, {oov_gain:.1f} percentage points above the smoothed HMM ({pct(hmm['OOV Accuracy'])}), and far above Brill TBL ({pct(brill['OOV Accuracy'])}) and N-gram back-off ({pct(ngram['OOV Accuracy'])}). Surprisingly, the hand-written regex tagger achieved {pct(regex['OOV Accuracy'])} on OOV tokens, outperforming both learned symbolic/statistical back-off models on this subset. Its suffix rules apply to any token, whereas Brill and N-gram models rely on lexical or contextual patterns seen during training and fall back to <code>NN</code> for truly unseen forms.
</div>
<p>This result is theoretically informative. Generalisation to unseen words is not automatically a property of "statistical" or "neural" models; it depends on whether the model contains a mechanism that operates on the internal structure of the word. The regex cascade has such a mechanism, albeit a coarse one. Brill TBL inherits the sparsity of its unigram baseline for OOV tokens and its learned transformations rarely fire reliably on unseen lexical items. The HMM's suffix interpolation provides a stronger mechanism by combining morphological evidence with transition probabilities. The BiLSTM learns a non-linear character composition function jointly with contextual encoding, which is why it is the strongest OOV tagger.</p>

<h3>5.3 Per-tag behaviour</h3>
{image('fig_per_tag_f1.png', 'Figure 3. Per-tag F1 for the 18 most frequent test tags. The neural model is consistently strong, while the regex tagger collapses on grammatical function words and ambiguous categories.')}
<p>Common nouns, punctuation, and determiners are tagged almost perfectly by all data-driven models. The larger differences appear on fine-grained verb and noun distinctions (<code>VBD</code> vs <code>VBN</code>, <code>NN</code> vs <code>NNP</code>, <code>JJ</code> vs <code>NN</code>) and on adverbs. These are precisely the categories where context and morphology matter, and where a first-order HMM or a suffix-only rule is insufficient.</p>

<h3>5.4 Confusion analysis</h3>
{image('fig_confusion_matrices.png', 'Figure 4. Row-normalised confusion matrices for the HMM and BiLSTM over the 15 most frequent tags. The neural diagonal is sharper for noun and verb categories; residual confusions reflect genuine ambiguity and annotation conventions.')}
<p>The confusion matrices show that errors are systematic rather than random. Both models confuse <code>NN</code> with <code>NNP</code> and <code>JJ</code>, and <code>VBD</code> with <code>VBN</code> and <code>VB</code>. The BiLSTM reduces many of these confusions because its bidirectional context representation can distinguish, for example, a participle used after an auxiliary from a finite past-tense verb. The remaining errors are often tokens that are ambiguous even for human annotators without sentence-level semantics.</p>

<h3>5.5 Efficiency and training dynamics</h3>
{image('fig_speed.png', 'Figure 5. Inference throughput on CPU, tagging one sentence at a time. Count-based and rule-based models are orders of magnitude faster than the neural model in this per-sentence API.')}
<p>The HMM tagged about {number(hmm['Tokens/Second'], 0)} tokens per second, roughly {speed_ratio:.1f} times faster than the BiLSTM ({number(neural['Tokens/Second'], 0)} tokens/s). Brill TBL and the regex tagger are faster still. The neural model's training time ({neural['Training Seconds']:.1f} s) is over three orders of magnitude larger than the HMM's ({hmm['Training Seconds']:.3f} s), although both are modest in absolute terms on this small corpus.</p>

{image('fig_training_time.png', 'Figure 6. Training cost by model on CPU. The BiLSTM dominates training time; the HMM and N-gram models require less than half a second.')}

{image('fig_training_curve.png', 'Figure 7. BiLSTM training dynamics. Validation accuracy rises quickly and plateaus; early stopping restores the best validation checkpoint.')}

<h3>5.6 Statistical significance</h3>
{table(
    ['Comparison', 'Only first correct', 'Only second correct', 'Discordant', 'McNemar p-value', 'Significant'],
    significance_rows,
    'Table 4. Exact McNemar tests on paired token-level predictions.',
    numeric_columns=(1, 2, 3),
)}
<p>All three comparisons are significant at the 0.05 level. The HMM versus BiLSTM comparison has {significance[0]['Discordant tokens']} discordant tokens, with the BiLSTM correct on {significance[0]['Only second model correct']} and the HMM correct on {significance[0]['Only first model correct']}. The Brill versus HMM comparison is also significant despite a smaller accuracy gap, and the Brill versus BiLSTM comparison has the strongest evidence of a difference. Statistical significance does not imply practical importance, but it confirms that the observed ordering is not an artefact of a handful of tokens.</p>

<h2>6. Discussion</h2>
<h3>6.1 Why the accuracy ordering arises</h3>
<p>Hand-written rules encode strong but local cues. They cannot represent the long tail of contextual ambiguity, so they plateau. Brill TBL learns contextual transformations from the training errors and therefore captures many local syntactic regularities, but it remains a symbolic learner over observed word and tag contexts; its OOV behaviour is weak because transformations are indexed by tags and words seen in training. The smoothed HMM adds a probabilistic treatment of uncertainty and a morphological fallback for unknown words. It is a strong baseline because POS sequences are highly regular: determiners precede nouns and adjectives, modals precede base verbs, and punctuation constrains sentence structure. The BiLSTM gains further because it learns continuous, context-dependent representations and composes characters into word representations, which is exactly what OOV tagging requires.</p>

<h3>6.2 The OOV surprise</h3>
<p>The regex tagger's advantage over Brill TBL on OOV tokens is a useful reminder that aggregate accuracy and OOV accuracy measure different capabilities. A model can be strong on frequent words and weak on rare ones, or vice versa. It also shows that "neural beats rules" is too coarse a statement: the relevant comparison is between specific mechanisms. Here, a simple morphological rule generalises better than a learned contextual rule cascade on unseen forms. The neural model dominates because it combines both mechanisms: character composition for unseen forms and bidirectional context for disambiguation.</p>

<h3>6.3 Trade-offs and practical recommendations</h3>
<ul>
  <li><strong>Use a rule-based tagger</strong> when training data is unavailable, transparency is mandatory, or the domain has stable, well-understood conventions. A hand-written cascade is also a valuable sanity-check baseline and can be maintained by domain experts.</li>
  <li><strong>Use Brill TBL</strong> when interpretable learned rules are required and a small annotated corpus is available. It offers a substantial accuracy gain over hand-written rules at negligible inference cost.</li>
  <li><strong>Use a smoothed HMM</strong> when a fast, lightweight, reasonably accurate tagger is needed and some OOV robustness is required. Suffix-smoothed emissions are a cheap and effective addition.</li>
  <li><strong>Use a character-aware BiLSTM</strong> when accuracy and OOV robustness are the priority and training/inference budgets permit. On this corpus it delivers the best overall and OOV results by a clear margin.</li>
  <li><strong>Do not choose a model by accuracy alone.</strong> Report OOV accuracy, macro F1, throughput, and training cost, because rankings can change depending on the deployment constraint.</li>
</ul>

<h3>6.4 Interpretation and transparency</h3>
<p>The rule-based and Brill models can be inspected rule by rule. The HMM exposes transition and emission tables, which support targeted debugging (for example, checking the probability of <code>DT &rarr; NN</code> or the emission of a suffix under <code>VBG</code>). The BiLSTM's behaviour can be probed with error analysis and confusion matrices, but its internal decision process is not directly readable. In applications where accountability matters (for example, educational or legal text processing), this difference may outweigh a few points of accuracy.</p>

<h2>7. Threats to Validity</h2>
<ul>
  <li><strong>Corpus size and domain.</strong> The NLTK Treebank sample is small and drawn from news and fiction. Absolute scores and rankings may differ on the full Penn Treebank, Universal Dependencies, or domain-specific text.</li>
  <li><strong>Single split.</strong> A fixed 80/10/10 split was used for reproducibility. Cross-validation or repeated random splits would provide confidence intervals and reduce split-specific variance.</li>
  <li><strong>No pretrained representations.</strong> The neural model was deliberately trained from scratch. Pretrained word embeddings or a fine-tuned transformer would likely raise accuracy, especially on rare words, at the cost of external dependencies and compute.</li>
  <li><strong>Timing methodology.</strong> Timings are single-run CPU measurements and include per-sentence Python overhead. Batched GPU inference would change the throughput comparison substantially.</li>
  <li><strong>Tagset conventions.</strong> The PTB tagset contains annotation distinctions that are not fully predictable from the token and its local context. Some residual errors are therefore irreducible without document-level information or annotation guidelines.</li>
  <li><strong>No CRF layer.</strong> The neural tagger uses independent softmax outputs and can in principle produce invalid tag sequences. A CRF or structured decoder could improve consistency.</li>
</ul>

<h2>8. Conclusion and Future Work</h2>
<p>This project compared rule-based, statistical, and neural POS tagging under a controlled protocol. The results show a clear accuracy progression from hand-written rules ({pct(regex['Accuracy'])}) to Brill TBL ({pct(brill['Accuracy'])}), to a smoothed HMM ({pct(hmm['Accuracy'])}), to a character-aware BiLSTM ({pct(neural['Accuracy'])}). The neural model also achieved the best OOV accuracy ({pct(neural['OOV Accuracy'])}), while the HMM provided the best accuracy-per-cost trade-off and the rule-based models provided the greatest transparency. The OOV comparison produced a nuanced finding: a simple morphological rule tagger generalised better to unseen words than a learned contextual rule cascade, which highlights the importance of mechanism-level analysis rather than paradigm-level labels.</p>
<p>Future work should add a CRF or structured attention layer, evaluate pretrained embeddings and transformer encoders, test on Universal Dependencies and domain-shifted corpora, report bootstrap confidence intervals, and explore distillation of the neural tagger into a compact HMM or rule cascade for low-resource deployment.</p>

<h2>References</h2>
<ol class="refs">
  <li>Brill, E. (1992). A simple rule-based part of speech tagger. <em>Proceedings of the Third Conference on Applied Natural Language Processing</em>, 152-155.</li>
  <li>Huang, Z., Xu, W., &amp; Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. <em>arXiv:1508.01991</em>.</li>
  <li>Jurafsky, D., &amp; Martin, J. H. (2023). <em>Speech and Language Processing</em> (3rd ed. draft). Chapters on sequence labelling and POS tagging.</li>
  <li>Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., &amp; Dyer, C. (2016). Neural architectures for named entity recognition. <em>NAACL-HLT</em>.</li>
  <li>Loper, E., &amp; Bird, S. (2002). NLTK: The Natural Language Toolkit. <em>Proceedings of the ACL Workshop on Effective Tools and Methodologies for Teaching NLP</em>.</li>
  <li>Marcus, M. P., Santorini, B., &amp; Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English: The Penn Treebank. <em>Computational Linguistics</em>, 19(2), 313-330.</li>
  <li>Plank, B., Sogaard, A., &amp; Goldberg, Y. (2016). Multilingual part-of-speech tagging with bidirectional long short-term memory models and auxiliary loss. <em>ACL</em>.</li>
</ol>

<h2>Appendix A. Repository Structure</h2>
<pre>POS-Tagging-Project/
├── notebooks/
│   └── POS_Tagging_Comparison.ipynb
├── src/
│   ├── data.py
│   ├── rule_based.py
│   ├── statistical.py
│   ├── neural.py
│   └── evaluate.py
├── scripts/
│   ├── build_notebook.py
│   └── build_report.py
├── assets/
│   ├── figures/
│   └── bilstm_pos.pt
├── reports/
│   ├── POS_Tagging_Report.pdf
│   ├── output_screenshots.pdf
│   ├── model_comparison.csv
│   └── results.json
├── data/README.md
├── requirements.txt
├── .gitignore
└── README.md</pre>

<h2>Appendix B. Reproduction</h2>
<pre>python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace \\
    notebooks/POS_Tagging_Comparison.ipynb
python scripts/build_report.py
</pre>
<p>The notebook downloads the corpus on first execution, trains all five models, saves the comparison artefacts, and regenerates every figure used in this report.</p>

<h2>Appendix C. Deliverables Mapping</h2>
<table class="data">
<thead><tr><th>Deliverable</th><th>Location</th></tr></thead>
<tbody>
<tr><td>Complete Jupyter notebook</td><td><code>notebooks/POS_Tagging_Comparison.ipynb</code></td></tr>
<tr><td>Project report (PDF)</td><td><code>reports/POS_Tagging_Report.pdf</code></td></tr>
<tr><td>Output screenshots (PDF)</td><td><code>reports/output_screenshots.pdf</code></td></tr>
<tr><td>Demo video (MP4)</td><td>Recorded separately; script in <code>reports/demo_video_script.md</code></td></tr>
<tr><td>GitHub repository</td><td><a href="https://github.com/vvkmlt/POS-Tagging-Project">https://github.com/vvkmlt/POS-Tagging-Project</a></td></tr>
</tbody>
</table>

</body>
</html>
"""

output = REPORT_DIR / "POS_Tagging_Report.html"
output.write_text(report_html, encoding="utf-8")
print(f"Wrote {output} ({output.stat().st_size:,} bytes)")

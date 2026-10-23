# Demo Video Script (2:45-3:00, MP4)

**Goal:** show the project objective, the three approaches, and the empirical results, including
the OOV comparison, in under three minutes.

**Tools:** OBS Studio (recommended), Loom, or Zoom. Record at 1080p, 30 fps. Export as
`demo_video.mp4` (H.264). Keep the notebook and the report figures open in separate tabs/windows.

**Before recording**

1. Run the notebook from top to bottom so all outputs and figures are visible.
2. Open `notebooks/POS_Tagging_Comparison.ipynb` in Jupyter and collapse long output cells.
3. Open `assets/figures/fig_model_accuracy.png` and `fig_per_tag_f1.png` in an image viewer.
4. Close notifications and unrelated tabs. Zoom the notebook to about 110% for readability.
5. Do one rehearsal with a timer; the target is 2:45.

---

## Shot list and narration

### 0:00-0:20 - Title and objective

**Screen:** title cell of the notebook (project title, author, course).

**Narration:**
> "Hello, I'm [Name], and this is my term project: a theoretical and empirical comparison of
> rule-based, statistical, and neural POS tagging. I implemented five taggers on the Penn Treebank
> and evaluated them on the same train/test split using accuracy, macro F1, OOV accuracy, training
> time, and inference speed."

### 0:20-0:45 - Dataset and protocol

**Screen:** EDA section: dataset summary, tag distribution figure, OOV statistics.

**Narration:**
> "The corpus has 3,914 sentences and 45 PTB tags. I use an 80/10/10 sentence-level split with a
> fixed seed. The test set has 9,995 tokens, including 721 out-of-vocabulary tokens. Every model
> is evaluated with exactly the same function, so the comparison is fair."

**Action:** briefly hover over the OOV rate (7.21%) and the tag distribution.

### 0:45-1:15 - Rule-based models

**Screen:** the regex rule table, a sample tagged sentence, then the Brill learned-rules table and
its metrics.

**Narration:**
> "The first family is rule-based. The hand-written tagger uses ordered punctuation, lexicon, and
> suffix rules, with no training. It reaches 73.9 percent accuracy, which shows the limits of
> hand-crafted context. Brill transformation-based learning starts from a unigram baseline and
> greedily learns contextual rewrite rules. Those rules are still readable, and they raise accuracy
> to 91.6 percent."

### 1:15-1:50 - Statistical models

**Screen:** HMM transition table, sample tagging, HMM classification report/accuracy, then the
N-gram back-off result.

**Narration:**
> "The second family is statistical. The bigram HMM models transition and emission probabilities,
> uses add-alpha smoothing, and decodes with Viterbi. Rare words are mapped to UNK, and unknown
> words get an interpolated suffix distribution. It trains in under a tenth of a second and reaches
> 93.0 percent accuracy. The N-gram back-off tagger is faster still but weaker, at 87 percent,
> because unseen words fall back to NN."

### 1:50-2:25 - Neural model

**Screen:** architecture markdown, training log, training curve, BiLSTM accuracy.

**Narration:**
> "The third family is neural. The model embeds words and characters, runs a character BiLSTM to
> compose morphology, then a word-level BiLSTM for bidirectional context. It trains on CPU for about
> three minutes with early stopping. It reaches 96.2 percent overall accuracy and 87.4 percent on
> OOV tokens, the best of all five models."

### 2:25-2:50 - Comparison and the key insight

**Screen:** comparison table, accuracy bar chart, per-tag heatmap, and the OOV callout.

**Narration:**
> "Here is the main comparison. Accuracy rises from hand-written rules, to Brill, to the HMM, to
> the neural model. The neural model cuts the HMM's error rate by about 46 percent relative and is
> 7 times slower per sentence, so the choice is a real accuracy-versus-cost trade-off. One
> surprising result: on OOV tokens the simple regex tagger beats Brill, 62 versus 24 percent,
> because suffix rules generalise to unseen words while Brill's contextual rules do not. That is
> the main lesson: OOV robustness depends on the mechanism, not the paradigm label."

### 2:50-3:00 - Conclusion and close

**Screen:** conclusion section or report cover.

**Narration:**
> "In summary, rules are transparent, HMMs offer excellent value, and character-aware neural
> taggers are the most accurate and OOV-robust but the most expensive. The full notebook, report,
> and outputs are in the GitHub repository. Thank you."

---

## Recording checklist

- [ ] 1080p, clear audio, no background noise.
- [ ] Cursor movements are slow and deliberate.
- [ ] Show at least one code block from each family.
- [ ] Show final metrics and at least one figure.
- [ ] Show the OOV comparison explicitly.
- [ ] Video length between 2:30 and 3:00.
- [ ] Export to `demo_video.mp4` and verify playback on another device.
- [ ] Upload to the GitHub repository or link it from the README (if file size allows; otherwise
      use a linked cloud drive and note the link in the README).

## Suggested screen layout

- Left 70%: Jupyter notebook.
- Right 30%: report figures or a text editor with the script (hidden during the final take).

## Optional B-roll

- 2 seconds of the repository tree in File Explorer.
- 2 seconds of `reports/output_screenshots.pdf` open at the comparison table.
- 2 seconds of `reports/results.json` for reproducibility evidence.

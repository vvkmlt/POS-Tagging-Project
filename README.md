# Rule-Based vs. Statistical vs. Neural POS Tagging

A controlled theoretical and empirical comparison of five Part-of-Speech (POS) taggers on the
Penn Treebank sample: a hand-written regex/lexicon tagger, Brill transformation-based learning,
a smoothed bigram HMM, an N-gram back-off tagger, and a character-aware BiLSTM.

The complete study, including exploratory data analysis, model implementations, evaluation,
figures, and error analysis, lives in
[`notebooks/POS_Tagging_Comparison.ipynb`](notebooks/POS_Tagging_Comparison.ipynb).

## Results at a glance

Test set: 9,995 tokens, 721 OOV tokens (7.21%), 45 PTB tags. All models use the same 80/10/10
sentence-level split (seed 42) and the same evaluation function.

| Model | Accuracy | Macro F1 | OOV accuracy | Train (s) | Tokens/s |
|---|---:|---:|---:|---:|---:|
| Neural (BiLSTM, char-aware) | **96.22%** | **90.96%** | **87.38%** | 186.0 | 4,651 |
| Statistical (Bigram HMM) | 93.01% | 89.59% | 68.24% | 0.09 | 36,885 |
| Rule-based (Brill TBL) | 91.62% | 90.22% | 23.72% | 3.49 | 233,053 |
| Statistical (N-gram back-off) | 87.00% | 86.67% | 18.72% | 0.38 | 898,863 |
| Rule-based (regex/lexicon) | 73.88% | 53.22% | 62.41% | 0.00 | 968,836 |

**Main findings**

- The accuracy ordering matches the expected evolution rules -> statistics -> neural, but a
  well-smoothed HMM is a strong low-cost baseline.
- The BiLSTM gives the best overall accuracy and a large OOV advantage, because character
  composition handles unseen morphology and bidirectional context resolves ambiguity.
- Brill TBL closes most of the gap between hand-written rules and statistical models while
  remaining human-readable.
- **Nuance:** the hand-written regex tagger beats Brill TBL and N-gram back-off on OOV accuracy.
  Its suffix rules apply to any token, while Brill's learned contextual rules depend on patterns
  observed during training. Generalisation to OOV words is a property of the *mechanism*, not of
  the paradigm label.
- The neural-vs-HMM and Brill-vs-HMM accuracy differences are statistically significant under an
  exact McNemar test (p < 0.001).

## Repository structure

```
POS-Tagging-Project/
├── notebooks/
│   └── POS_Tagging_Comparison.ipynb   # complete executed study
├── src/
│   ├── data.py                        # corpus loading and 80/10/10 split
│   ├── rule_based.py                  # regex/lexicon tagger and Brill TBL
│   ├── statistical.py                 # smoothed HMM + Viterbi, N-gram back-off
│   ├── neural.py                      # character-aware BiLSTM (PyTorch)
│   └── evaluate.py                    # metrics, OOV accuracy, error analysis
├── scripts/
│   ├── build_notebook.py              # regenerates the notebook from source cells
│   ├── build_report.py                # builds the report HTML from results.json
│   └── make_pdfs.ps1                  # rebuilds the report and screenshots PDFs
├── assets/
│   ├── figures/                       # all plots used in the report
│   └── bilstm_pos.pt                  # trained neural checkpoint (after execution)
├── reports/
│   ├── POS_Tagging_Report.pdf         # term project report
│   ├── output_screenshots.pdf         # executed-notebook outputs as PDF
│   ├── output_screenshots.html
│   ├── demo_video_script.md           # MP4 recording script and shot list
│   ├── model_comparison.csv
│   └── results.json
├── data/README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python 3.10+ is recommended (the results in this repository were produced with Python 3.14,
PyTorch 2.9 CPU, NLTK 3.10).

```powershell
cd E:\TA
python -m pip install -r requirements.txt
python -m jupyter notebook notebooks\POS_Tagging_Comparison.ipynb
```

Run all cells from top to bottom. The Penn Treebank sample is downloaded automatically by NLTK on
first use; no manual data download is required.

The notebook takes roughly 4-6 minutes on a modern CPU. Most of that time is BiLSTM training
(about 190 seconds for 12 epochs with early stopping).

## Reproducing the artifacts

```powershell
# Re-execute the notebook and regenerate results/figures (in place)
python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=2400 notebooks\POS_Tagging_Comparison.ipynb

# Rebuild the formatted report HTML + PDF and the outputs PDF
powershell -ExecutionPolicy Bypass -File scripts\make_pdfs.ps1
```

`scripts/make_pdfs.ps1` uses Microsoft Edge in headless mode to print the HTML files to PDF. If
Edge is installed elsewhere, pass `-EdgePath "C:\path\to\msedge.exe"`.

The notebook is generated from `scripts/build_notebook.py`. Edit that script (not the `.ipynb`
directly) if you want to change narrative text or code cells, then run:

```powershell
python scripts\build_notebook.py
```

## Notebook outline

1. Introduction and objectives
2. Experimental setup and reproducibility
3. Dataset preparation and EDA (tag distribution, sentence lengths, OOV statistics)
4. Rule-based tagging: regex/lexicon cascade and Brill TBL with learned rules
5. Statistical tagging: bigram HMM with add-alpha smoothing and suffix-aware OOV emissions,
   plus an N-gram back-off baseline
6. Neural tagging: character-aware BiLSTM, training curves, early stopping
7. Comparative analysis: accuracy, macro F1, OOV accuracy, throughput, per-tag F1,
   confusion matrices, error galleries, McNemar significance tests
8. Discussion, threats to validity, conclusion, future work
9. Reproducibility notes and artifact index

## Deliverables

| Deliverable | Location |
|---|---|
| Jupyter notebook | `notebooks/POS_Tagging_Comparison.ipynb` |
| Project report (PDF) | `reports/POS_Tagging_Report.pdf` |
| Output screenshots (PDF) | `reports/output_screenshots.pdf` |
| Demo video (MP4) | Record using `reports/demo_video_script.md` |
| GitHub repository | Publish this folder (see below) |

Before submission, replace the placeholders in the report cover and notebook title cell:
`<Your Name>`, `<Your Roll Number>`, `<Course Name>`, `<Institution Name>`, and
`<GitHub Repository URL>`.

## Publish to GitHub

```powershell
cd E:\TA
# A local git repository (branch main) is already initialized.
git add .
git commit -m "Rule-based, statistical, and neural POS tagging comparison"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

The trained neural checkpoint and temporary HTML/PDF files are excluded by `.gitignore`; the
notebook regenerates the checkpoint. The report and screenshots PDFs are intentionally committed
because they are graded deliverables.

## Implementation notes

- **Rule-based tagger:** ordered cascade of punctuation/irregular lookups, closed-class lexicons,
  and morphological regexes, defaulting to `NN`. No training data is used.
- **Brill TBL:** unigram baseline backed off to `NN`, `nltkdemo18` templates, maximum 200 rules.
- **HMM:** bigram transitions, add-alpha smoothing, rare-word `<UNK>` mapping, suffix-conditioned
  emissions for OOV tokens, Viterbi decoding.
- **N-gram:** trigram -> bigram -> unigram -> `NN` back-off, cutoff 1.
- **BiLSTM:** learned word embeddings (100-d), character BiLSTM (2 x 50) with max-pooling,
  word BiLSTM (2 x 128), dropout 0.33, Adam, gradient clipping, early stopping. No pretrained
  embeddings, so the experiment is fully self-contained.

## References

- Brill, E. (1992). A simple rule-based part of speech tagger. *ANLP*.
- Marcus, Santorini, & Marcinkiewicz (1993). Building a large annotated corpus of English: The
  Penn Treebank. *Computational Linguistics*.
- Jurafsky & Martin (2023). *Speech and Language Processing* (3rd ed. draft).
- Lample et al. (2016). Neural architectures for named entity recognition. *NAACL-HLT*.
- Plank, Sogaard, & Goldberg (2016). Multilingual POS tagging with BiLSTMs and auxiliary loss. *ACL*.

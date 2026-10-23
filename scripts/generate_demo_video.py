"""Generate a narrated slide-based demo video from the project results.

This produces a reproducible MP4 walkthrough (1080p) using Windows SAPI for
narration and the locally installed figures/results. Replace it with your own
screen recording if your course requires a live demo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "assets" / "figures"
RESULTS = json.loads((REPORT_DIR / "results.json").read_text(encoding="utf-8"))
BY_MODEL = {row["Model"]: row for row in RESULTS["results"]}

W, H = 1920, 1080
NAVY = "#12263A"
DARK = "#0E2030"
ACCENT = "#C9A227"
BG = "#F4F7FA"
TEXT = "#1F2933"
MUTED = "#5B6770"
BORDER = "#DDE3EA"
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"
WHITE = "#FFFFFF"

FONT_DIR = Path(r"C:\Windows\Fonts")
AUTHOR = "VENKATESH V  |  E720524AM034"
INSTITUTION = "EASA COLLEGE OF ENGINEERING AND TECHNOLOGY"
REPO = "github.com/vvkmlt/POS-Tagging-Project"

NARRATION = [
    (
        "title",
        "Hello, I am Venkatesh V, and this is my term project: a theoretical and empirical "
        "comparison of rule-based, statistical, and neural part-of-speech tagging. I implemented "
        "five taggers on the Penn Treebank and evaluated them on the same test split.",
    ),
    (
        "dataset",
        "The corpus contains 3,914 sentences annotated with 45 Penn Treebank tags. I use an "
        "eighty-ten-ten sentence-level split with a fixed seed. The test set has 9,995 tokens, "
        "including 721 out-of-vocabulary tokens.",
    ),
    (
        "regex",
        "The first family is rule-based. The hand-written tagger applies punctuation, closed-class "
        "lexicon, and suffix rules, with no training. For example, words ending in I-N-G become "
        "V-B-G, and capitalised words become N-N-P. It reaches 73.9 percent accuracy, showing the "
        "limits of hand-crafted context.",
    ),
    (
        "brill",
        "Brill transformation-based learning starts from a unigram baseline and greedily learns "
        "contextual rewrite rules that reduce training error. The rules remain human-readable, and "
        "accuracy rises to 91.6 percent. However, its OOV accuracy is only 23.7 percent because the "
        "learned rules depend on contexts seen during training.",
    ),
    (
        "hmm",
        "The second family is statistical. The bigram hidden Markov model estimates transition and "
        "emission probabilities, applies add-alpha smoothing, and decodes with Viterbi. Rare words "
        "map to U-N-K, and unknown words receive a suffix-based emission distribution. It trains in "
        "under a tenth of a second and reaches 93 percent accuracy.",
    ),
    (
        "ngram",
        "A trigram tagger with back-off provides a second statistical baseline. It is extremely "
        "fast, but unseen words fall back to the default noun tag, so overall accuracy is 87 percent "
        "and OOV accuracy only 18.7 percent.",
    ),
    (
        "neural",
        "The third family is neural. The model combines word embeddings with a character BiLSTM "
        "that composes morphology, followed by a word-level BiLSTM for bidirectional context. It "
        "reaches 96.2 percent overall accuracy and 87.4 percent on OOV tokens.",
    ),
    (
        "results",
        "Here is the complete comparison. Accuracy rises from hand-written rules, to Brill, to the "
        "smoothed HMM, to the neural model. The BiLSTM reduces the HMM error rate by about 46 "
        "percent relative, but it is roughly eight times slower per sentence.",
    ),
    (
        "oov",
        "One finding is particularly interesting. On OOV tokens, the simple regex tagger beats "
        "Brill, 62 versus 24 percent, because suffix rules apply to any word, while Brill's "
        "contextual rules do not generalise to unseen forms. OOV robustness depends on the "
        "mechanism, not the paradigm label.",
    ),
    (
        "conclusion",
        "In summary, rules are transparent, HMMs offer excellent value, and character-aware neural "
        "taggers are the most accurate and the most robust to unseen words, at a higher cost. The "
        "full notebook, report, and outputs are available in the GitHub repository. Thank you.",
    ),
]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


def bold(size: int) -> ImageFont.FreeTypeFont:
    return font("segoeuib.ttf", size)


def regular(size: int) -> ImageFont.FreeTypeFont:
    return font("segoeui.ttf", size)


def mono(size: int) -> ImageFont.FreeTypeFont:
    return font("consola.ttf", size)


def wrap(draw: ImageDraw.ImageDraw, text: str, text_font, max_width: int):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=text_font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def wrapped_text(draw, x, y, text, text_font, fill, max_width, line_gap=8):
    lines = wrap(draw, text, text_font, max_width)
    height = draw.textbbox((0, 0), "Ag", font=text_font)[3]
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        y += height + line_gap
    return y


def bullets(draw, x, y, items, text_font, max_width, color=TEXT, gap=20, accent=ACCENT):
    for item in items:
        draw.ellipse([x, y + 11, x + 12, y + 23], fill=accent)
        y = wrapped_text(draw, x + 30, y, item, text_font, color, max_width - 30, line_gap=6)
        y += gap
    return y


def header(draw, title: str, number: int, kicker="POS TAGGING COMPARISON"):
    draw.rectangle([0, 0, W, 120], fill=NAVY)
    draw.rectangle([0, 116, W, 120], fill=ACCENT)
    draw.text((90, 18), kicker, font=bold(18), fill=ACCENT)
    draw.text((90, 46), title, font=bold(40), fill=WHITE)
    label = f"{number:02d} / {len(NARRATION):02d}"
    width = draw.textlength(label, font=bold(20))
    draw.text((W - 90 - width, 54), label, font=bold(20), fill="#8FA6B8")


def footer(draw):
    draw.line([90, 1000, W - 90, 1000], fill=BORDER, width=2)
    draw.text((90, 1018), f"{AUTHOR}  |  {INSTITUTION}", font=regular(17), fill=MUTED)
    width = draw.textlength(REPO, font=regular(17))
    draw.text((W - 90 - width, 1018), REPO, font=regular(17), fill=MUTED)


def metric_card(draw, x, y, width, height, value, label, accent, note=None):
    draw.rounded_rectangle(
        [x, y, x + width, y + height], radius=16, fill=WHITE, outline=BORDER, width=2
    )
    draw.rounded_rectangle(
        [x, y, x + 10, y + height], radius=5, fill=accent
    )
    draw.text((x + 32, y + 22), value, font=bold(52), fill=accent)
    draw.text((x + 34, y + 88), label, font=regular(21), fill=MUTED)
    if note:
        draw.text((x + 34, y + 116), note, font=regular(18), fill=MUTED)


def callout(draw, x, y, width, height, text, accent=GREEN, text_font=None, fill="#F3F7F4"):
    text_font = text_font or regular(24)
    draw.rounded_rectangle(
        [x, y, x + width, y + height], radius=14, fill=fill, outline=BORDER, width=2
    )
    draw.rectangle([x, y, x + 8, y + height], fill=accent)
    wrapped_text(draw, x + 28, y + 20, text, text_font, TEXT, width - 52, line_gap=6)


def paste_figure(canvas, path: Path, box, border=True):
    x0, y0, x1, y1 = box
    image = Image.open(path).convert("RGB")
    image.thumbnail((x1 - x0, y1 - y0), Image.LANCZOS)
    x = x0 + (x1 - x0 - image.width) // 2
    y = y0 + (y1 - y0 - image.height) // 2
    if border:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [x - 3, y - 3, x + image.width + 3, y + image.height + 3],
            outline=BORDER,
            width=2,
        )
    canvas.paste(image, (x, y))


def draw_table(draw, x, y, widths, headers, rows, row_height=50, text_size=22):
    total_width = sum(widths)
    header_height = 52
    total_height = header_height + row_height * len(rows)
    draw.rectangle([x, y, x + total_width, y + total_height], fill=WHITE, outline=BORDER, width=2)
    draw.rectangle([x, y, x + total_width, y + header_height], fill=NAVY)
    cx = x
    for heading, width in zip(headers, widths):
        draw.text((cx + 14, y + 14), heading, font=bold(20), fill=WHITE)
        cx += width
    for index, row in enumerate(rows):
        ry = y + header_height + index * row_height
        if index % 2:
            draw.rectangle([x + 2, ry, x + total_width - 2, ry + row_height], fill="#F2F5F8")
        cx = x
        for column, (cell, width) in enumerate(zip(row, widths)):
            text = str(cell)
            if column == 0:
                draw.text((cx + 14, ry + 13), text, font=regular(text_size), fill=TEXT)
            else:
                text_width = draw.textlength(text, font=bold(text_size))
                draw.text(
                    (cx + width - 14 - text_width, ry + 13),
                    text,
                    font=bold(text_size),
                    fill=TEXT,
                )
            cx += width


def box_arrow(draw, x, y, width, height, label, sublabel=None, accent=BLUE):
    draw.rounded_rectangle(
        [x, y, x + width, y + height], radius=12, fill=WHITE, outline=BORDER, width=2
    )
    draw.rectangle([x, y, x + width, y + 6], fill=accent)
    label_font = bold(23)
    label_width = draw.textlength(label, font=label_font)
    draw.text((x + (width - label_width) / 2, y + 16), label, font=label_font, fill=TEXT)
    if sublabel:
        sub_font = regular(17)
        sub_width = draw.textlength(sublabel, font=sub_font)
        draw.text((x + (width - sub_width) / 2, y + 48), sublabel, font=sub_font, fill=MUTED)


def down_arrow(draw, cx, y1, y2, color=NAVY):
    draw.line([cx, y1, cx, y2 - 8], fill=color, width=4)
    draw.polygon([(cx - 10, y2 - 12), (cx + 10, y2 - 12), (cx, y2)], fill=color)


def right_arrow(draw, x1, x2, cy, color=NAVY):
    draw.line([x1, cy, x2 - 8, cy], fill=color, width=4)
    draw.polygon([(x2 - 12, cy - 10), (x2 - 12, cy + 10), (x2, cy)], fill=color)


def slide_title():
    image = Image.new("RGB", (W, H), DARK)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, W, 14], fill=ACCENT)
    draw.ellipse([1430, -260, 2050, 360], fill="#16324F")
    draw.ellipse([-180, 760, 420, 1360], fill="#152B40")

    draw.text((110, 170), "TERM PROJECT  |  NATURAL LANGUAGE PROCESSING", font=bold(22), fill=ACCENT)
    title = "A Theoretical and Empirical Comparison of Rule-Based, Statistical, and Neural POS Tagging"
    y = 230
    for line in wrap(draw, title, bold(64), 1450):
        draw.text((110, y), line, font=bold(64), fill=WHITE)
        y += 78
    y += 18
    subtitle = (
        "Five taggers. One held-out split. Accuracy, OOV robustness, speed, training cost, "
        "and interpretability."
    )
    wrapped_text(draw, 110, y, subtitle, regular(30), "#AFC0CF", 1400, line_gap=8)

    draw.line([110, 760, 1810, 760], fill="#2D455C", width=2)
    draw.text((110, 800), "VENKATESH V", font=bold(34), fill=WHITE)
    draw.text((110, 848), "Roll No. E720524AM034  |  B.E - CSE (AI & ML), U24DSP13", font=regular(24), fill="#AFC0CF")
    draw.text((110, 888), INSTITUTION, font=regular(23), fill="#AFC0CF")
    draw.text((110, 950), REPO, font=bold(22), fill=ACCENT)
    return image


def slide_dataset():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Dataset and Experimental Protocol", 2)

    card_width = 550
    for index, (value, label) in enumerate(
        [
            ("3,914", "sentences in the corpus"),
            ("45", "Penn Treebank tags"),
            ("721", "OOV test tokens (7.21%)"),
        ]
    ):
        metric_card(draw, 90 + index * 585, 170, card_width, 145, value, label, [BLUE, ORANGE, RED][index])

    bullets(
        draw,
        100,
        370,
        [
            "Sentence-level 80/10/10 split with a fixed seed of 42; validation is used only for neural early stopping.",
            "9,995 test tokens and 704 OOV word types; every model sees the same test sentences.",
            "Identical token-level evaluation function for all five taggers.",
        ],
        regular(25),
        820,
    )

    paste_figure(image, FIG_DIR / "fig_tag_distribution.png", (100, 570, 1820, 960))
    footer(draw)
    return image


def slide_regex():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Approach 1 - Rule-Based Tagging (Hand-Written)", 3)

    draw.text((100, 175), "Ordered linguistic cascade", font=bold(30), fill=NAVY)
    bullets(
        draw,
        110,
        235,
        [
            "Punctuation and irregular-form lexicon lookups.",
            "Closed-class word lists: pronouns, determiners, prepositions, modals.",
            "Morphological regexes for suffixes, numbers, and capitalisation.",
            "No learned parameters; default tag is NN.",
        ],
        regular(24),
        780,
    )

    draw.rounded_rectangle([100, 540, 860, 900], radius=16, fill=DARK)
    draw.text((130, 560), "rule excerpt", font=bold(20), fill=ACCENT)
    rule_lines = [
        "-ing          ->  VBG",
        "-ed           ->  VBD",
        "-ly           ->  RB",
        "^[A-Z]        ->  NNP",
        "capitalised+s ->  NNPS",
        "default       ->  NN",
    ]
    for index, line in enumerate(rule_lines):
        draw.text((130, 610 + index * 44), line, font=mono(24), fill="#DCE6EE")

    metric_card(draw, 1010, 190, 790, 150, "73.88%", "overall test accuracy", BLUE)
    metric_card(draw, 1010, 365, 790, 150, "62.41%", "accuracy on OOV tokens", ORANGE)
    metric_card(draw, 1010, 540, 790, 150, "0 s", "training time (no learning)", GREEN)
    callout(
        draw,
        1010,
        725,
        790,
        155,
        "Transparent and free, but it cannot resolve contextual ambiguity: many tokens default to NN.",
        accent=RED,
        fill="#FBF1F1",
        text_font=regular(22),
    )
    footer(draw)
    return image


def slide_brill():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Approach 1b - Brill Transformation-Based Learning", 4)

    draw.text((100, 175), "Learned, human-readable rules", font=bold(30), fill=NAVY)
    bullets(
        draw,
        110,
        235,
        [
            "Start from a unigram baseline backed off to NN.",
            "Enumerate contextual transformations and greedily apply the best error-reducing rule.",
            "Learn 200 rules in about 3.5 seconds.",
            "Keeps symbolic transparency while adapting to the corpus.",
        ],
        regular(24),
        950,
    )

    metric_card(draw, 1210, 190, 590, 145, "91.62%", "overall test accuracy", BLUE)
    metric_card(draw, 1210, 355, 590, 145, "23.72%", "accuracy on OOV tokens", RED)
    metric_card(draw, 1210, 520, 590, 145, "200", "learned transformation rules", GREEN)

    draw.rounded_rectangle([100, 690, 1120, 950], radius=16, fill=DARK)
    draw.text((130, 708), "sample learned transformations", font=bold(19), fill=ACCENT)
    rules = [
        "NN->VB      if Pos:TO@[-1]",
        "VBP->VB     if Pos:MD@[-3,-2,-1]",
        "VBD->VBN    if Pos:VBZ@[-2,-1]",
        "VB->NN      if Pos:DT@[-1]",
        "VB->VBP     if Pos:NNS@[-1]",
        "VBN->VBD    if Pos:PRP@[-1]",
    ]
    for index, line in enumerate(rules):
        draw.text((130, 750 + index * 32), line, font=mono(20), fill="#DCE6EE")

    callout(
        draw,
        1210,
        715,
        590,
        205,
        "Learned context helps overall accuracy, but rules indexed by observed words and tags do not transfer to unseen word forms.",
        accent=ORANGE,
        fill="#FBF5EE",
        text_font=regular(21),
    )
    footer(draw)
    return image


def slide_hmm():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Approach 2 - Bigram Hidden Markov Model", 5)

    formula = "P(w, t)  =  product over i of  P(t_i | t_i-1)  x  P(w_i | t_i)"
    formula_font = font("cambria.ttc", 36)
    formula_width = draw.textlength(formula, font=formula_font)
    draw.text(((W - formula_width) / 2, 160), formula, font=formula_font, fill=NAVY)

    for index, (value, label, accent) in enumerate(
        [
            ("93.01%", "overall test accuracy", BLUE),
            ("0.09 s", "training time", GREEN),
            ("36,885", "tokens tagged per second", ORANGE),
        ]
    ):
        metric_card(draw, 95 + index * 590, 245, 555, 140, value, label, accent)

    bullets(
        draw,
        110,
        430,
        [
            "Add-alpha smoothing for start, transition, and emission distributions.",
            "Rare training words map to <UNK>, learning a mass for unknown words.",
            "OOV emissions interpolate the UNK distribution with a final-3-character suffix distribution (weight 0.35).",
            "Viterbi decoding in log space, O(n x T squared).",
        ],
        regular(23),
        860,
        gap=14,
    )

    draw.text((1090, 430), "example tag path", font=bold(22), fill=NAVY)
    tags = ["DT", "NN", "VBZ", "VBG", "NN"]
    box_width, box_height, gap = 120, 76, 35
    start_x = 1090
    for index, tag in enumerate(tags):
        bx = start_x + index * (box_width + gap)
        draw.rounded_rectangle(
            [bx, 480, bx + box_width, 480 + box_height], radius=12, fill=WHITE, outline=BORDER, width=2
        )
        draw.rectangle([bx, 480, bx + box_width, 486], fill=BLUE)
        width = draw.textlength(tag, font=bold(28))
        draw.text((bx + (box_width - width) / 2, 498), tag, font=bold(28), fill=TEXT)
        if index < len(tags) - 1:
            right_arrow(draw, bx + box_width + 5, bx + box_width + gap - 5, 518)

    callout(
        draw,
        1090,
        610,
        730,
        195,
        "Suffix-aware emissions raise OOV accuracy to 68.2%, far above plain N-gram back-off.",
        accent=GREEN,
        text_font=regular(22),
    )
    footer(draw)
    return image


def slide_ngram():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Approach 2b - N-Gram Back-Off Baseline", 6)

    for index, (value, label, accent) in enumerate(
        [
            ("87.00%", "overall test accuracy", BLUE),
            ("18.72%", "accuracy on OOV tokens", RED),
            ("898,863", "tokens tagged per second", GREEN),
        ]
    ):
        metric_card(draw, 95 + index * 590, 190, 555, 145, value, label, accent)

    bullets(
        draw,
        110,
        390,
        [
            "Trigram tagger backs off to bigram, then unigram, then the default tag NN.",
            "Back-off avoids zero probabilities but has no morphological model.",
            "Fastest count-based baseline, yet unseen word forms collapse to the default noun tag.",
        ],
        regular(25),
        880,
    )

    draw.text((1310, 330), "back-off hierarchy", font=bold(22), fill=NAVY)
    levels = [("Trigram", 420), ("Bigram", 340), ("Unigram", 260), ("Default NN", 180)]
    for index, (label, width) in enumerate(levels):
        y = 385 + index * 125
        x = 1320 + (420 - width) / 2
        draw.rounded_rectangle(
            [x, y, x + width, y + 82], radius=12, fill=WHITE, outline=BORDER, width=2
        )
        draw.rectangle([x, y, x + width, y + 6], fill=[BLUE, ORANGE, GREEN, RED][index])
        text_width = draw.textlength(label, font=bold(24))
        draw.text((x + (width - text_width) / 2, y + 26), label, font=bold(24), fill=TEXT)
        if index < len(levels) - 1:
            down_arrow(draw, 1530, y + 82, y + 125 - 6)

    callout(
        draw,
        100,
        700,
        1080,
        150,
        "This baseline isolates the value of smoothing and morphology: removing them costs about 6 accuracy points and 50 OOV points versus the HMM.",
        accent=ORANGE,
        fill="#FBF5EE",
        text_font=regular(22),
    )
    footer(draw)
    return image


def slide_neural():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Approach 3 - Character-Aware BiLSTM", 7)

    draw.text((100, 175), "Architecture", font=bold(30), fill=NAVY)
    boxes = [
        ("Word embedding", "100 dimensions, learned from scratch", BLUE),
        ("Character BiLSTM", "2 x 50 hidden units, max-pooled", ORANGE),
        ("Concatenation", "word vector + character vector", GREEN),
        ("Word BiLSTM", "2 x 128 hidden units, bidirectional context", PURPLE),
        ("Linear + softmax", "45 tag scores per token", RED),
    ]
    for index, (label, sublabel, accent) in enumerate(boxes):
        y = 235 + index * 128
        box_arrow(draw, 100, y, 760, 88, label, sublabel, accent)
        if index < len(boxes) - 1:
            down_arrow(draw, 480, y + 88, y + 128 - 4)

    paste_figure(image, FIG_DIR / "fig_training_curve.png", (960, 175, 1825, 745))

    metric_card(draw, 95, 800, 555, 130, "96.22%", "overall test accuracy", BLUE)
    metric_card(draw, 680, 800, 555, 130, "87.38%", "accuracy on OOV tokens", GREEN)
    metric_card(draw, 1265, 800, 555, 130, "186 s", "CPU training (early stop)", ORANGE)
    footer(draw)
    return image


def slide_results():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Empirical Comparison", 8)

    paste_figure(image, FIG_DIR / "fig_model_accuracy.png", (90, 165, 1120, 690))

    rows = []
    for name, label in [
        ("Neural (BiLSTM)", "BiLSTM"),
        ("Statistical (HMM)", "HMM"),
        ("Rule-Based (Brill TBL)", "Brill TBL"),
        ("Statistical (N-gram Backoff)", "N-gram"),
        ("Rule-Based (Regex)", "Regex"),
    ]:
        row = BY_MODEL[name]
        rows.append(
            [
                label,
                f"{row['Accuracy'] * 100:.2f}",
                f"{row['OOV Accuracy'] * 100:.2f}",
            ]
        )

    draw_table(
        draw,
        1160,
        175,
        [250, 130, 130],
        ["Model", "Acc %", "OOV %"],
        rows,
        row_height=52,
        text_size=21,
    )

    callout(
        draw,
        1160,
        560,
        650,
        230,
        "BiLSTM vs HMM: 96.2% vs 93.0% accuracy, but 8x slower inference and roughly 2,000x the training cost.",
        accent=GREEN,
        text_font=regular(21),
    )

    draw.text(
        (100, 740),
        "Accuracy ordering: rules < Brill < HMM < neural",
        font=bold(28),
        fill=NAVY,
    )
    draw.text(
        (100, 790),
        "Macro F1 tells a subtler story: Brill (90.2%) nearly matches the BiLSTM (91.0%) on rare tags.",
        font=regular(23),
        fill=MUTED,
    )
    footer(draw)
    return image


def slide_oov():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "OOV Accuracy: The Decisive Difference", 9)

    paste_figure(image, REPORT_DIR / "_oov_chart.png", (90, 170, 1040, 820))

    draw.text((1100, 190), "Why the regex tagger beats Brill", font=bold(28), fill=NAVY)
    bullets(
        draw,
        1110,
        250,
        [
            "Suffix rules apply to any token, including unseen words.",
            "Brill's learned rules depend on contexts observed during training.",
            "HMM suffix smoothing improves OOV to 68.2%.",
            "The character BiLSTM composes unseen words and reaches 87.4%.",
        ],
        regular(23),
        680,
        gap=14,
    )

    callout(
        draw,
        1090,
        690,
        730,
        185,
        "OOV robustness depends on the mechanism, not the paradigm label.",
        accent=RED,
        fill="#FBF1F1",
        text_font=bold(24),
    )
    footer(draw)
    return image


def slide_conclusion():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(draw, "Conclusion", 10)

    cards = [
        ("Rule-based", "73.9% / 91.6%", "Regex / Brill TBL", "Transparent, zero or low training cost", BLUE),
        ("Statistical HMM", "93.0%", "0.09 s training", "Best accuracy-per-cost trade-off", ORANGE),
        ("Neural BiLSTM", "96.2% / 87.4%", "Accuracy / OOV", "Best overall and open-vocabulary performance", GREEN),
    ]
    for index, (title, value, label, note, accent) in enumerate(cards):
        x = 95 + index * 590
        draw.rounded_rectangle(
            [x, 185, x + 555, 520], radius=18, fill=WHITE, outline=BORDER, width=2
        )
        draw.rectangle([x, 185, x + 555, 197], fill=accent)
        draw.text((x + 30, 225), title, font=bold(30), fill=NAVY)
        draw.text((x + 30, 285), value, font=bold(46), fill=accent)
        draw.text((x + 30, 355), label, font=regular(20), fill=MUTED)
        wrapped_text(draw, x + 30, 405, note, regular(22), TEXT, 490, line_gap=6)

    draw.text(
        (100, 590),
        "Use rules for transparency, HMMs for efficiency, neural models for open-vocabulary accuracy.",
        font=bold(32),
        fill=NAVY,
    )

    draw.line([100, 690, 1820, 690], fill=BORDER, width=2)
    draw.text((100, 730), "Full notebook, report, and results", font=bold(26), fill=NAVY)
    draw.text((100, 775), REPO, font=bold(30), fill=ACCENT)
    draw.text((100, 860), "Thank you", font=bold(56), fill=NAVY)
    footer(draw)
    return image


BUILDERS = {
    "title": slide_title,
    "dataset": slide_dataset,
    "regex": slide_regex,
    "brill": slide_brill,
    "hmm": slide_hmm,
    "ngram": slide_ngram,
    "neural": slide_neural,
    "results": slide_results,
    "oov": slide_oov,
    "conclusion": slide_conclusion,
}


def make_oov_chart(target: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["N-gram", "Brill TBL", "Regex", "HMM", "BiLSTM"]
    labels = {
        "N-gram": "Statistical (N-gram Backoff)",
        "Brill TBL": "Rule-Based (Brill TBL)",
        "Regex": "Rule-Based (Regex)",
        "HMM": "Statistical (HMM)",
        "BiLSTM": "Neural (BiLSTM)",
    }
    values = [BY_MODEL[labels[name]]["OOV Accuracy"] * 100 for name in order]
    colors = [ORANGE, ORANGE, BLUE, BLUE, GREEN]

    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=170)
    bars = ax.barh(order, values, color=colors, height=0.62)
    ax.set_xlim(0, 100)
    ax.set_xlabel("OOV accuracy (%)", fontsize=11)
    ax.set_title("Accuracy on out-of-vocabulary test tokens", fontsize=12, fontweight="bold")
    for bar, value in zip(bars, values):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def synthesize(text: str, target: Path, work: Path, index: int):
    text_file = work / f"narration_{index:02d}.txt"
    text_file.write_text(text, encoding="utf-8")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "tts.ps1"),
            "-TextFile",
            str(text_file),
            "-OutFile",
            str(target),
            "-Voice",
            "Microsoft Zira Desktop",
            "-Rate",
            "1",
        ],
        check=True,
        capture_output=True,
    )


def pad_wav(source: Path, target: Path, lead=0.10, tail=0.32) -> float:
    with wave.open(str(source), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    silence = b"\x00" * (params.sampwidth * params.nchannels)
    lead_frames = silence * int(params.framerate * lead)
    tail_frames = silence * int(params.framerate * tail)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(params.nchannels)
        handle.setsampwidth(params.sampwidth)
        handle.setframerate(params.framerate)
        handle.writeframes(lead_frames + frames + tail_frames)
    return (len(lead_frames) + len(frames) + len(tail_frames)) / (
        params.sampwidth * params.nchannels * params.framerate
    )


def concat_wavs(paths, target: Path) -> float:
    total_frames = 0
    with wave.open(str(target), "wb") as output:
        params = None
        for path in paths:
            with wave.open(str(path), "rb") as handle:
                if params is None:
                    params = handle.getparams()
                    output.setnchannels(params.nchannels)
                    output.setsampwidth(params.sampwidth)
                    output.setframerate(params.framerate)
                assert handle.getparams()[:3] == params[:3]
                frames = handle.readframes(handle.getnframes())
                output.writeframes(frames)
                total_frames += handle.getnframes()
    return total_frames / (params.framerate)


def main():
    work = Path(tempfile.mkdtemp(prefix="pos_demo_", dir=str(REPORT_DIR)))
    try:
        make_oov_chart(REPORT_DIR / "_oov_chart.png")

        audio_files = []
        durations = []
        for index, (kind, text) in enumerate(NARRATION, start=1):
            raw_wav = work / f"raw_{index:02d}.wav"
            padded_wav = work / f"audio_{index:02d}.wav"
            synthesize(text, raw_wav, work, index)
            durations.append(pad_wav(raw_wav, padded_wav))
            audio_files.append(padded_wav)

            slide = BUILDERS[kind]()
            slide.save(work / f"slide_{index:02d}.png", "PNG")
            print(f"Built slide {index:02d} ({kind}) - {durations[-1]:.1f}s")

        narration = work / "narration.wav"
        total_audio = concat_wavs(audio_files, narration)

        slides_file = work / "slides.txt"
        lines = []
        for index, duration in enumerate(durations, start=1):
            lines.append(f"file '{(work / f'slide_{index:02d}.png').as_posix()}'")
            lines.append(f"duration {duration:.3f}")
        lines.append(f"file '{(work / f'slide_{len(durations):02d}.png').as_posix()}'")
        slides_file.write_text("\n".join(lines), encoding="utf-8")

        output = REPORT_DIR / "demo_video.mp4"
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(slides_file),
            "-i",
            str(narration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "stillimage",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-shortest",
            "-t",
            f"{total_audio:.3f}",
            str(output),
        ]
        subprocess.run(command, check=True, capture_output=True)
        print(f"Wrote {output} ({output.stat().st_size / 1_000_000:.1f} MB)")
        print(f"Audio duration: {total_audio:.1f} s")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        chart = REPORT_DIR / "_oov_chart.png"
        if chart.exists():
            chart.unlink()


if __name__ == "__main__":
    main()

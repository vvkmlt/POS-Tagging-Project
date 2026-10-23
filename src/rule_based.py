"""Rule-based POS taggers: a hand-written regex tagger and Brill TBL."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from nltk.tag import BrillTaggerTrainer, DefaultTagger, UnigramTagger
from nltk.tag.brill import nltkdemo18

from .data import Sentence

Rule = Tuple[str, str]

CLOSED_CLASS: dict[str, str] = {}

CLOSED_CLASS.update(
    {
        word: "PRP"
        for word in (
            "i me myself we us ourselves you yourself yourselves he him himself "
            "she her herself it itself they them themselves one oneself"
        ).split()
    }
)
CLOSED_CLASS.update(
    {
        word: "PRP$"
        for word in "my our your his her its their mine ours yours hers theirs".split()
    }
)
CLOSED_CLASS.update(
    {
        word: "DT"
        for word in (
            "the a an this that these those some any every no another each "
            "either neither such all both half"
        ).split()
    }
)
CLOSED_CLASS.update(
    {
        word: "IN"
        for word in (
            "of in on at by for with about against between into through during "
            "before after above below to from up down off over under again "
            "because while although though since unless until upon within without "
            "across along among around behind beyond despite except inside near "
            "outside past toward towards via"
        ).split()
    }
)
CLOSED_CLASS.update(
    {word: "CC" for word in "and or but nor yet so if whether".split()}
)
CLOSED_CLASS.update(
    {word: "MD" for word in "can could may might must shall should will would".split()}
)
CLOSED_CLASS.update(
    {word: "WP" for word in "who whom whose which what whatever whoever".split()}
)
CLOSED_CLASS.update(
    {word: "WRB" for word in "when where why how whenever wherever".split()}
)
CLOSED_CLASS.update(
    {
        word: "RB"
        for word in (
            "not never very too also just only even still already always often "
            "sometimes soon now then today tomorrow yesterday well quite rather "
            "almost enough here there once twice however therefore thus hence "
            "meanwhile instead perhaps maybe indeed ago"
        ).split()
    }
)
CLOSED_CLASS.update({"to": "TO"})
CLOSED_CLASS.update(
    {
        word: "UH"
        for word in "oh ah wow hello hi yes yeah okay ok please thanks".split()
    }
)
CLOSED_CLASS.update(
    {
        word: "VBZ" for word in "is has does".split()
    }
)
CLOSED_CLASS.update(
    {
        word: "VBP" for word in "am are have do".split()
    }
)
CLOSED_CLASS.update(
    {
        word: "VBD" for word in "was were had did".split()
    }
)
CLOSED_CLASS.update(
    {
        word: "VB" for word in "be have do".split()
    }
)
CLOSED_CLASS.update(
    {
        word: "VBN" for word in "been done gone seen taken given written".split()
    }
)
CLOSED_CLASS.update(
    {
        word: "VBG" for word in "being having doing going".split()
    }
)

IRREGULAR: dict[str, str] = {
    "said": "VBD",
    "says": "VBZ",
    "went": "VBD",
    "made": "VBD",
    "took": "VBD",
    "came": "VBD",
    "knew": "VBD",
    "thought": "VBD",
    "gave": "VBD",
    "found": "VBD",
    "told": "VBD",
    "became": "VBD",
    "left": "VBD",
    "felt": "VBD",
    "got": "VBD",
    "children": "NNS",
    "men": "NNS",
    "women": "NNS",
    "people": "NNS",
    "feet": "NNS",
    "teeth": "NNS",
    "mice": "NNS",
    "data": "NNS",
    "indices": "NNS",
    "mr.": "NNP",
    "mrs.": "NNP",
    "ms.": "NNP",
    "dr.": "NNP",
}

PUNCTUATION: dict[str, str] = {
    ",": ",",
    ".": ".",
    "!": ".",
    "?": ".",
    ";": ":",
    ":": ":",
    "...": ":",
    "``": "``",
    "''": "''",
    "`": "``",
    "(": "-LRB-",
    ")": "-RRB-",
    "[": "-LRB-",
    "]": "-RRB-",
    "{": "-LRB-",
    "}": "-RRB-",
    "#": "#",
    "$": "$",
    "%": "NN",
    "&": "CC",
}

REGEX_RULES: List[Rule] = [
    (r"^[.,;:!?]+$", "."),
    (r"^[-A-Za-z0-9/]+\.$", "NNP"),
    (r"^[A-Z]{2,}$", "NNP"),
    (r"^[A-Z][a-z]+$", "NNP"),
    (r"^-?\$?\d[\d,.]*$", "CD"),
    (r"^[A-Za-z]+ly$", "RB"),
    (r"^[A-Za-z]+ing$", "VBG"),
    (r"^[A-Za-z]+ed$", "VBD"),
    (r"^[A-Za-z]+(tion|sion|ment|ness|ity|ance|ence|ship|hood|ism|ist)$", "NN"),
    (r"^[A-Za-z]+(ous|ful|able|ible|ive|al|ic|ary|less|ish)$", "JJ"),
    (r"^[A-Za-z]+(ss|us|is)$", "NN"),
    (r"^[A-Za-z]+(s|es)$", "NNS"),
]


class RegexRuleTagger:
    """A deterministic tagger driven only by hand-written rules and lexicons."""

    name = "Rule-Based (Regex)"

    def __init__(self, default_tag: str = "NN") -> None:
        self.default_tag = default_tag
        self.rules = [(re.compile(pattern), tag) for pattern, tag in REGEX_RULES]

    def tag(self, tokens: Sequence[str]) -> List[str]:
        tagged: List[str] = []
        for token in tokens:
            tagged.append(self._tag_token(token))
        return tagged

    def _tag_token(self, token: str) -> str:
        if token in PUNCTUATION:
            return PUNCTUATION[token]

        lowered = token.lower()
        if lowered in IRREGULAR:
            return IRREGULAR[lowered]
        if lowered in CLOSED_CLASS:
            return CLOSED_CLASS[lowered]
        if token.isdigit():
            return "CD"

        for pattern, tag in self.rules:
            if pattern.match(token):
                return tag
        return self.default_tag


class BrillRuleTagger:
    """Transformation-based learning: a learned, human-readable rule cascade."""

    name = "Rule-Based (Brill TBL)"

    def __init__(
        self,
        max_rules: int = 200,
        min_score: int = 2,
        trace: int = 0,
        default_tag: str = "NN",
    ) -> None:
        self.max_rules = max_rules
        self.min_score = min_score
        self.trace = trace
        self.default_tag = default_tag
        self._tagger = None

    def fit(self, train: Sequence[Sentence]) -> "BrillRuleTagger":
        baseline = UnigramTagger(
            list(train), backoff=DefaultTagger(self.default_tag)
        )
        trainer = BrillTaggerTrainer(
            baseline, nltkdemo18(), deterministic=True, trace=self.trace
        )
        self._tagger = trainer.train(
            list(train), max_rules=self.max_rules, min_score=self.min_score
        )
        return self

    def tag(self, tokens: Sequence[str]) -> List[str]:
        if self._tagger is None:
            raise RuntimeError("Call fit() before tag()")
        return [tag for _, tag in self._tagger.tag(list(tokens))]

    def learned_rules(self) -> List[str]:
        if self._tagger is None:
            return []
        return [str(rule) for rule in self._tagger.rules()]

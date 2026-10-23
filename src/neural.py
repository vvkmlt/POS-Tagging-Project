"""A character-aware BiLSTM POS tagger implemented in PyTorch."""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import Sentence

PAD = "<pad>"
UNK = "<unk>"
MAX_CHARS = 16


@dataclass
class Vocabs:
    word2idx: Dict[str, int]
    char2idx: Dict[str, int]
    tag2idx: Dict[str, int]
    idx2tag: List[str]
    max_chars: int = MAX_CHARS

    @property
    def n_words(self) -> int:
        return len(self.word2idx)

    @property
    def n_chars(self) -> int:
        return len(self.char2idx)

    @property
    def n_tags(self) -> int:
        return len(self.tag2idx)


def build_vocabs(
    train: Sequence[Sentence],
    min_word_freq: int = 2,
    min_char_freq: int = 1,
    max_chars: int = MAX_CHARS,
) -> Vocabs:
    word_counts = Counter(word for sentence in train for word, _ in sentence)
    char_counts = Counter(ch for sentence in train for word, _ in sentence for ch in word)
    tagset = sorted({tag for sentence in train for _, tag in sentence})

    word2idx = {PAD: 0, UNK: 1}
    for word, count in sorted(word_counts.items()):
        if count >= min_word_freq:
            word2idx[word] = len(word2idx)

    char2idx = {PAD: 0, UNK: 1}
    for char, count in sorted(char_counts.items()):
        if count >= min_char_freq:
            char2idx[char] = len(char2idx)

    tag2idx = {tag: index for index, tag in enumerate(tagset)}
    return Vocabs(word2idx, char2idx, tag2idx, tagset, max_chars)


class TaggingDataset(Dataset):
    def __init__(self, sentences: Sequence[Sentence], vocabs: Vocabs) -> None:
        self.samples = []
        for sentence in sentences:
            words = [word for word, _ in sentence]
            tags = [tag for _, tag in sentence]
            word_ids = torch.tensor(
                [vocabs.word2idx.get(word, vocabs.word2idx[UNK]) for word in words],
                dtype=torch.long,
            )
            char_ids = torch.zeros(len(words), vocabs.max_chars, dtype=torch.long)
            for i, word in enumerate(words):
                chars = [vocabs.char2idx.get(ch, vocabs.char2idx[UNK]) for ch in word[: vocabs.max_chars]]
                char_ids[i, : len(chars)] = torch.tensor(chars, dtype=torch.long)
            tag_ids = torch.tensor([vocabs.tag2idx[tag] for tag in tags], dtype=torch.long)
            self.samples.append((word_ids, char_ids, tag_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate_batch(batch):
    max_len = max(len(word_ids) for word_ids, _, _ in batch)
    batch_size = len(batch)
    max_chars = batch[0][1].shape[1]

    words = torch.zeros(batch_size, max_len, dtype=torch.long)
    chars = torch.zeros(batch_size, max_len, max_chars, dtype=torch.long)
    tags = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i, (word_ids, char_ids, tag_ids) in enumerate(batch):
        length = len(word_ids)
        words[i, :length] = word_ids
        chars[i, :length] = char_ids
        tags[i, :length] = tag_ids
    return words, chars, tags


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        n_words: int,
        n_chars: int,
        n_tags: int,
        word_dim: int = 100,
        char_dim: int = 30,
        char_hidden: int = 50,
        hidden: int = 128,
        dropout: float = 0.33,
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(
            char_dim, char_hidden, batch_first=True, bidirectional=True
        )
        self.word_lstm = nn.LSTM(
            word_dim + 2 * char_hidden,
            hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, n_tags)

    def forward(self, words: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = words.shape
        flat_chars = chars.view(batch_size * seq_len, -1)
        char_embeddings = self.char_embedding(flat_chars)
        char_outputs, _ = self.char_lstm(char_embeddings)
        char_mask = flat_chars.ne(0).unsqueeze(-1)
        char_outputs = char_outputs.masked_fill(~char_mask, -1e4)
        char_features = char_outputs.max(dim=1).values
        char_features = char_features.view(batch_size, seq_len, -1)

        word_features = self.word_embedding(words)
        features = torch.cat([word_features, char_features], dim=-1)
        features = self.dropout(features)
        outputs, _ = self.word_lstm(features)
        outputs = self.dropout(outputs)
        return self.classifier(outputs)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _accuracy(model: BiLSTMTagger, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for words, chars, tags in loader:
            words, chars, tags = words.to(device), chars.to(device), tags.to(device)
            predictions = model(words, chars).argmax(dim=-1)
            mask = tags.ne(-100)
            correct += int((predictions[mask] == tags[mask]).sum())
            total += int(mask.sum())
    return correct / max(total, 1)


def train_bilstm(
    train_sents: Sequence[Sentence],
    val_sents: Sequence[Sentence],
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[BiLSTMTagger, Vocabs, dict]:
    set_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    vocabs = build_vocabs(train_sents)
    torch_device = torch.device(device)

    train_loader = DataLoader(
        TaggingDataset(train_sents, vocabs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TaggingDataset(val_sents, vocabs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = BiLSTMTagger(
        vocabs.n_words, vocabs.n_chars, vocabs.n_tags
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    history = {"epoch": [], "train_loss": [], "val_accuracy": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        token_count = 0
        for words, chars, tags in train_loader:
            words, chars, tags = words.to(torch_device), chars.to(torch_device), tags.to(torch_device)
            optimizer.zero_grad()
            logits = model(words, chars)
            loss = criterion(logits.view(-1, vocabs.n_tags), tags.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach()) * int(tags.ne(-100).sum())
            token_count += int(tags.ne(-100).sum())

        val_accuracy = _accuracy(model, val_loader, torch_device)
        train_loss = running_loss / max(token_count, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)

        if verbose:
            print(
                f"epoch {epoch:02d} | train loss {train_loss:.4f} | "
                f"val accuracy {val_accuracy:.4f}"
            )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    model.eval()
    history["best_val_accuracy"] = best_val_accuracy
    return model, vocabs, history


class NeuralTagger:
    name = "Neural (BiLSTM)"

    def __init__(self, model: BiLSTMTagger, vocabs: Vocabs, device: str = "cpu") -> None:
        self.model = model
        self.vocabs = vocabs
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def tag(self, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []
        word_ids = torch.tensor(
            [[self.vocabs.word2idx.get(word, self.vocabs.word2idx[UNK]) for word in tokens]],
            dtype=torch.long,
            device=self.device,
        )
        char_ids = torch.zeros(
            1, len(tokens), self.vocabs.max_chars, dtype=torch.long, device=self.device
        )
        for i, word in enumerate(tokens):
            chars = [
                self.vocabs.char2idx.get(ch, self.vocabs.char2idx[UNK])
                for ch in word[: self.vocabs.max_chars]
            ]
            char_ids[0, i, : len(chars)] = torch.tensor(
                chars, dtype=torch.long, device=self.device
            )
        with torch.no_grad():
            logits = self.model(word_ids, char_ids)
            indices = logits.argmax(dim=-1)[0].tolist()
        return [self.vocabs.idx2tag[index] for index in indices]

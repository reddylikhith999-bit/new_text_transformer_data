"""Train a Transformer sentiment classifier for fastText review files.

Example:
    python train.py --train-file train.ft.txt --test-file test.ft.txt
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
LABEL_RE = re.compile(r"^(__label__[12])\s+(.*)$")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def read_reviews(file_path, max_samples=None):
    """Yield (review, label) pairs from a fastText-format file."""
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = LABEL_RE.match(line.rstrip("\r\n"))
            if not match:
                continue
            label, review = match.groups()
            yield review, 0 if label == "__label__1" else 1
            count += 1
            if max_samples is not None and count >= max_samples:
                break


def build_vocab(file_path, max_vocab, max_samples=None, min_freq=1):
    counts = Counter()
    for review, _ in read_reviews(file_path, max_samples):
        counts.update(tokenize(review))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, frequency in counts.most_common():
        if frequency < min_freq or len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, max_length):
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]
    ids = ids[:max_length] or [vocab[UNK_TOKEN]]
    return torch.tensor(ids, dtype=torch.long)


def batches(file_path, vocab, batch_size, max_length, max_samples=None):
    texts, labels = [], []
    for text, label in read_reviews(file_path, max_samples):
        texts.append(encode(text, vocab, max_length))
        labels.append(label)
        if len(texts) == batch_size:
            yield texts, torch.tensor(labels, dtype=torch.long)
            texts, labels = [], []
    if texts:
        yield texts, torch.tensor(labels, dtype=torch.long)


def collate(texts, pad_id=0):
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(
        texts, batch_first=True, padding_value=pad_id
    )
    return padded, lengths


class SentimentTransformer(nn.Module):
    """Transformer encoder with sinusoidal positions and masked mean pooling."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=128,
        dropout=0.3,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, 2)

    def positional_encoding(self, length, device, dtype):
        position = torch.arange(
            length, device=device, dtype=dtype
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.embedding_dim)
        )

        angles = position * div_term
        encoding = torch.zeros(
            length, self.embedding_dim, device=device, dtype=dtype
        )
        encoding[:, 0::2] = torch.sin(angles)

        odd_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(angles[:, :odd_width])

        return encoding.unsqueeze(0)

    def forward(self, tokens, lengths=None):
        padding_mask = tokens.eq(0)

        embedded = self.embedding(tokens) * math.sqrt(self.embedding_dim)
        embedded = embedded + self.positional_encoding(
            tokens.size(1), embedded.device, embedded.dtype
        )

        encoded = self.encoder(
            embedded,
            src_key_padding_mask=padding_mask,
        )

        valid_tokens = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * valid_tokens).sum(dim=1)
        pooled = pooled / valid_tokens.sum(dim=1).clamp_min(1)

        return self.classifier(self.dropout(pooled))


# Keeps existing app.py imports working. Retrain before using the new model.
SentimentLSTM = SentimentTransformer


def evaluate(model, file_path, vocab, args, device):
    actual, predicted = [], []
    model.eval()

    with torch.no_grad():
        for texts, labels in batches(
            file_path,
            vocab,
            args.batch_size,
            args.max_length,
            args.max_test_samples,
        ):
            tokens, lengths = collate(texts)
            logits = model(tokens.to(device), lengths.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())

    if actual:
        print(f"Test accuracy: {accuracy_score(actual, predicted):.4f}")
        print(
            classification_report(
                actual,
                predicted,
                target_names=["negative", "positive"],
                zero_division=0,
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer for Amazon review sentiment."
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file")
    parser.add_argument("--model-out", default="sentiment_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if args.test_file and not Path(args.test_file).is_file():
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    if min(
        args.batch_size,
        args.epochs,
        args.max_vocab,
        args.max_length,
        args.num_heads,
        args.num_layers,
    ) < 1:
        parser.error("batch size, epochs, vocab, max length, heads, and layers must be positive")

    if args.embedding_dim % args.num_heads:
        parser.error("--embedding-dim must be divisible by --num-heads")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building vocabulary from {train_path}...")

    vocab = build_vocab(
        train_path,
        args.max_vocab,
        args.max_train_samples,
        args.min_freq,
    )
    print(f"Vocabulary size: {len(vocab):,}; device: {device}")

    model = SentimentTransformer(
        len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss, seen = 0.0, 0

        for texts, labels in batches(
            train_path,
            vocab,
            args.batch_size,
            args.max_length,
            args.max_train_samples,
        ):
            tokens, _ = collate(texts)
            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(tokens), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(labels)
            seen += len(labels)

        average_loss = running_loss / max(seen, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {average_loss:.4f}")

    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_type": "transformer",
            "model_state_dict": model.cpu().state_dict(),
            "vocab": vocab,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_length": args.max_length,
            "labels": {"0": "negative", "1": "positive"},
        },
        output_path,
    )
    print(f"Saved model to {output_path}")

    if args.test_file:
        evaluate(model.to(device), args.test_file, vocab, args, device)


if __name__ == "__main__":
    main()
    
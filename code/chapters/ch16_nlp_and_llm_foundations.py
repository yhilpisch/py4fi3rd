"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 16 - NLP and LLM Foundations.

This companion module provides small, runnable utilities that mirror the
chapter's core workflows: tokenization, sparse text vectors, baseline
classification, and a toy self-attention implementation.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - optional dependency
    CountVectorizer = None
    TfidfVectorizer = None
    LogisticRegression = None
    accuracy_score = None
    train_test_split = None


ArrayF = npt.NDArray[np.float64]


def simple_tokenize(text: str) -> list[str]:
    """Lower-case and split a string into simple alphanumeric tokens."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return [tok for tok in text.split() if tok]


def build_vocab(tokens: Iterable[str]) -> dict[str, int]:
    """Build a stable token-to-id mapping."""

    vocab = sorted(set(tokens))
    return {tok: i for i, tok in enumerate(vocab)}


def encode(tokens: Sequence[str], mapping: Mapping[str, int]) -> list[int]:
    """Encode tokens as integer IDs using a provided vocabulary mapping."""

    return [mapping[tok] for tok in tokens]


def softmax(x: ArrayF, axis: int = -1) -> ArrayF:
    """Numerically stable softmax."""

    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


@dataclass(frozen=True)
class AttentionResult:
    attention: ArrayF
    output: ArrayF


def toy_self_attention(
    tokens: Sequence[str],
    d_model: int = 8,
    seed: int = 2027,
) -> AttentionResult:
    """Compute single-head self-attention for a toy token sequence.

    The function uses random embeddings and random projection matrices to keep
    the example reproducible and dependency-free.
    """

    if d_model <= 0:
        raise ValueError("d_model must be positive.")
    if len(tokens) == 0:
        raise ValueError("Need at least one token.")

    rng = np.random.default_rng(seed=seed)
    X = rng.normal(scale=0.5, size=(len(tokens), d_model)).astype(np.float64)
    W_q = rng.normal(scale=0.5, size=(d_model, d_model)).astype(np.float64)
    W_k = rng.normal(scale=0.5, size=(d_model, d_model)).astype(np.float64)
    W_v = rng.normal(scale=0.5, size=(d_model, d_model)).astype(np.float64)

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = (Q @ K.T) / math.sqrt(d_model)
    attn = softmax(scores, axis=-1)
    Z = attn @ V
    return AttentionResult(attention=attn, output=Z)


def require_sklearn() -> None:
    """Raise a clear error if scikit-learn is unavailable."""

    if CountVectorizer is None or TfidfVectorizer is None:
        raise RuntimeError(
            "scikit-learn is required for the chapter 16 helpers."
        )


@dataclass(frozen=True)
class TextModelScore:
    accuracy: float
    vocab_size: int


def tfidf_logreg_demo(seed: int = 2027) -> TextModelScore:
    """Train a small TF-IDF + logistic-regression baseline on toy headlines."""

    require_sklearn()
    assert train_test_split is not None
    assert LogisticRegression is not None
    assert accuracy_score is not None

    texts = [
        "Stocks rise after strong earnings",
        "Shares jump as guidance surprises to the upside",
        "Bond yields fall on recession fears",
        "Market dips on inflation concerns",
        "Energy sector rallies on higher oil prices",
        "Tech shares retreat after weak outlook",
        "Bank stocks slide as rates outlook shifts",
        "Index climbs as risk appetite returns",
        "Credit spreads widen amid volatility",
        "Equities drop after cautious central bank comments",
        "Safe-haven demand lifts government bonds",
        "Stocks advance as macro data beats expectations",
    ]
    # 1 = broadly positive/risk-on; 0 = negative/risk-off (synthetic labels)
    y = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.33,
        random_state=seed,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train_vec, y_train)
    pred = clf.predict(X_test_vec)
    acc = float(accuracy_score(y_test, pred))
    return TextModelScore(
        accuracy=acc,
        vocab_size=int(len(vectorizer.vocabulary_)),
    )


def main() -> None:
    """Run compact chapter-16 demos with basic self-checks."""

    headlines = [
        "AAPL beats earnings expectations, shares jump",
        "Bank stocks slide as rates outlook shifts",
        "Energy sector rallies on higher oil prices",
        "Tech shares retreat after strong year-to-date gains",
    ]
    toks = [simple_tokenize(h) for h in headlines]
    assert all(isinstance(t, list) for t in toks)
    assert all(all(isinstance(x, str) for x in t) for t in toks)

    all_tokens = [tok for seq in toks for tok in seq]
    vocab = build_vocab(all_tokens)
    assert len(vocab) > 0
    enc = [encode(seq, vocab) for seq in toks]
    assert len(enc) == len(headlines)
    assert all(len(e) == len(t) for e, t in zip(enc, toks))

    attn_res = toy_self_attention(
        ["rates", "cut", "to", "boost", "equities"],
        d_model=8,
    )
    assert attn_res.attention.shape[0] == attn_res.attention.shape[1]
    assert np.allclose(attn_res.attention.sum(axis=1), 1.0)
    assert attn_res.output.shape[0] == attn_res.attention.shape[0]

    if CountVectorizer is not None:
        score = tfidf_logreg_demo()
        assert 0.0 <= score.accuracy <= 1.0
        assert score.vocab_size > 0
        print(score)


if __name__ == "__main__":
    main()

"""Answer generation on top of retrieved passages.

Default mode is extractive: MedQuAD answers are already self-contained
curated passages, so the practical, fully-offline, reproducible choice is to
return the top re-ranked passage's answer text directly. This keeps the
whole pipeline runnable without any external API key.

An optional LLM-synthesis mode is available (`--use-llm`, requires
`OPENAI_API_KEY`) that asks a chat model to compose a short answer from the
top-k retrieved passages. It silently falls back to the extractive answer if
the API call fails for any reason (no key, no network, quota, etc.).
"""
import os
import time
from dataclasses import dataclass
from typing import List

import pandas as pd

import config


@dataclass
class GeneratedAnswer:
    text: str
    mode: str
    latency_ms: float


def _extractive_answer(context_docs: pd.DataFrame) -> str:
    if len(context_docs) == 0:
        return ""
    return str(context_docs.iloc[0]["answer"])


def _llm_answer(question: str, context_docs: pd.DataFrame) -> str:
    from openai import OpenAI

    client = OpenAI()
    context = "\n\n".join(
        f"[{row.focus_area}] {row.answer}" for row in context_docs.itertuples()
    )
    prompt = (
        "Answer the medical question using ONLY the context passages below. "
        "Be concise and factual.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def generate_answer(question: str, context_docs: pd.DataFrame, use_llm: bool = False) -> GeneratedAnswer:
    t0 = time.perf_counter()

    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            text = _llm_answer(question, context_docs)
            latency_ms = (time.perf_counter() - t0) * 1000
            return GeneratedAnswer(text=text, mode="llm", latency_ms=latency_ms)
        except Exception as exc:  # pragma: no cover - network/optional path
            print(f"[qa_pipeline] LLM generation failed ({exc}); falling back to extractive answer")

    text = _extractive_answer(context_docs)
    latency_ms = (time.perf_counter() - t0) * 1000
    return GeneratedAnswer(text=text, mode="extractive", latency_ms=latency_ms)

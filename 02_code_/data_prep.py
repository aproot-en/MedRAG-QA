"""Load and clean the MedQuAD disease-QA subset (01_data/medquad.csv).

Cleaning steps applied (see 01_data analysis notes):
  - drop rows with missing `answer` or `focus_area` (this is what removes the
    CDC "What is (are)  ?" template-scrape rows, which have no focus_area)
  - drop exact duplicate (question, answer) rows
  - drop empty/near-empty question text (a safety net; removes no rows on
    the current medquad.csv)
  - collapse whitespace and strip "(Watch the video ...)" style boilerplate.
    The non-greedy regex stops at the first ")", so notes containing
    "(Esc)" leave a " button on your keyboard.)" fragment in 58
    NIHSeniorHealth answers. Kept as-is so the cached index and all
    reported results stay reproducible; fixing it requires --rebuild.
  - optionally cap the corpus size for a lightweight demo run
"""
import re
import pandas as pd

import config

_WHITESPACE_RE = re.compile(r"\s+")
_VIDEO_NOTE_RE = re.compile(r"\(Watch the video.*?\)", re.IGNORECASE | re.DOTALL)


def _clean_text(text: str) -> str:
    text = _VIDEO_NOTE_RE.sub(" ", str(text))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_clean_corpus(max_docs: int = config.MAX_DOCS, seed: int = config.RANDOM_SEED,
                       chunk_chars: int = config.DOC_CHUNK_CHARS) -> pd.DataFrame:
    """Return a cleaned, de-duplicated corpus with a stable `doc_id` column."""
    df = pd.read_csv(config.DATA_PATH)

    df = df.dropna(subset=["answer", "focus_area"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].str.len() > 5]  # safety net for empty questions

    df["question"] = df["question"].map(_clean_text)
    df["answer"] = df["answer"].map(_clean_text)
    df["focus_area"] = df["focus_area"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    if max_docs is not None and max_docs > 0 and len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df["doc_id"] = df.index.astype(int)
    # Text actually embedded/indexed: entity name + ANSWER only (question is
    # deliberately excluded). Retrieval must match a natural-language
    # question against the answer passage that addresses it -- the standard
    # QA-retrieval formulation. Including the question text here would let a
    # query trivially match its own document by literal string overlap and
    # make the retrieval/answer-quality evaluation meaningless.
    df["doc_text"] = (df["focus_area"] + ". " + df["answer"]).str.slice(0, chunk_chars)

    return df


if __name__ == "__main__":
    corpus = load_clean_corpus()
    print(f"Cleaned corpus: {len(corpus)} documents, {corpus['focus_area'].nunique()} unique focus areas, "
          f"{corpus['source'].nunique()} sources")
    print(corpus[["doc_id", "question", "focus_area", "source"]].head())

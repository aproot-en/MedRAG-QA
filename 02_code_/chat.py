"""Interactive terminal chat with the GraphRAG + BGE-M3 medical QA pipeline.

`run_experiment.py` only runs the fixed batch evaluation and exits -- it was
never meant to be interactive. This script loads the same cached index and
opens a simple read-question / print-answer loop instead.

Usage:
    python 02_code/chat.py
    python 02_code/chat.py --max-docs 5000   # match a different cached build
    python 02_code/chat.py --use-llm         # synthesize with OpenAI instead of extractive
    python 02_code/chat.py --show-context     # also print the retrieved passages

Type a question and press Enter. Type "exit", "quit" or Ctrl+C to stop.
"""
import argparse

import config
from qa_pipeline import generate_answer
from retriever import build_or_load_index


def parse_args():
    p = argparse.ArgumentParser(description="Chat with the GraphRAG + BGE-M3 medical QA pipeline")
    p.add_argument("--max-docs", type=int, default=config.MAX_DOCS,
                    help="corpus size to load from cache (default: config.MAX_DOCS, i.e. the full dataset)")
    p.add_argument("--top-k", type=int, default=config.FINAL_TOP_K)
    p.add_argument("--use-llm", action="store_true",
                    help="synthesize answers with an OpenAI chat model (requires OPENAI_API_KEY); "
                         "falls back to extractive answers otherwise")
    p.add_argument("--show-context", action="store_true", help="print the retrieved passages before the answer")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading GraphRAG + BGE-M3 index (uses the cached build if available)...")
    retriever, corpus, graph, embedder = build_or_load_index(force_rebuild=False, max_docs=args.max_docs)
    print(f"Ready. Corpus: {len(corpus)} documents, {corpus['focus_area'].nunique()} entities.")
    print('Type a medical question and press Enter. Type "exit" or "quit" to stop.\n')

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        retrieval = retriever.retrieve(question, final_top_k=args.top_k)
        context_docs = retriever.get_docs(retrieval.doc_ids)
        answer = generate_answer(question, context_docs, use_llm=args.use_llm)

        if args.show_context:
            print("\n--- retrieved passages ---")
            for rank, row in enumerate(context_docs.itertuples(), start=1):
                print(f"[{rank}] ({row.focus_area}, {row.source}) {row.answer[:200]}...")
            print("--------------------------")

        print(f"\nRAG [{answer.mode}]: {answer.text}\n")


if __name__ == "__main__":
    main()

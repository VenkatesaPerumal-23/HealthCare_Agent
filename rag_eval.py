"""
RAG retrieval calibration & evaluation for HealthGuard AI.

Run from the project folder (same place as config.py, retrieval.py, graph.py,
main.py, chroma_db/, .env):

  1) See real score distributions before picking a threshold:
       python rag_eval.py inspect
  2) Measure Recall@K, Precision@K, MRR across thresholds:
       python rag_eval.py evaluate --testset rag_testset.json
     add --with-coverage to include the LLM coverage check (slower, uses API calls)
  3) Create a starter test set to edit:
       python rag_eval.py template

Test-set format (pages are 1-based, as printed in the PDF viewer; page is optional):
[
  {"question": "How is dengue transmitted?",
   "expected": [{"source": "dengue_guidelines.pdf", "page": 4}]},
  {"question": "What is the capital of France?", "expected": []}   <- negative: should return nothing
]
"""
import argparse
import json
import statistics
import warnings

warnings.filterwarnings("ignore")

import config
import retrieval

K = config.RETRIEVAL_K
SWEEP = [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def is_relevant(source: dict, expected: list) -> bool:
    """A retrieved source counts as relevant if filename (and page, when given) match."""
    for item in expected:
        if source.get("source") != item.get("source"):
            continue
        if item.get("page") is None:
            return True
        page = source.get("page")
        if page is not None and int(page) + 1 == int(item["page"]):
            return True
    return False


def score_query(sources: list, expected: list, k: int) -> dict:
    top = sources[:k]
    hits = [is_relevant(s, expected) for s in top]
    first_hit = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    return {
        "recall": 1.0 if first_hit else 0.0,          # did ANY expected source appear in top-K
        "precision": (sum(hits) / len(top)) if top else 0.0,
        "rr": (1.0 / first_hit) if first_hit else 0.0,
        "returned": len(top),
    }


def cmd_inspect(args):
    queries = args.queries or [
        # On-topic: edit to match your PDFs
        "How is dengue transmitted?",
        "What are the symptoms of malaria?",
        "How can tuberculosis be prevented?",
        # Off-topic: these show what 'irrelevant' scores look like in YOUR store
        "What is the capital of France?",
        "How do I change a car tyre?",
    ]
    print(f"Raw cosine scores (before dedup/threshold), fetch_k={retrieval.RETRIEVAL_FETCH_K}\n")
    for query in queries:
        candidates = retrieval.retrieve_candidates(query)
        scores = [c["score"] for c in candidates]
        print(f"Q: {query}")
        for c in candidates[:6]:
            meta = c["doc"].metadata
            page = meta.get("page")
            page_txt = f"p{int(page) + 1}" if page is not None else "-"
            snippet = " ".join(c["doc"].page_content.split())[:70]
            print(f"   {c['score']:.3f}  {meta.get('source', '?')} {page_txt}  | {snippet}")
        if scores:
            print(f"   max={max(scores):.3f} median={statistics.median(scores):.3f} min={min(scores):.3f}\n")
    print("Pick a threshold BELOW the scores of clearly relevant chunks and ABOVE the")
    print("best scores of off-topic queries. Confirm it with `evaluate` before using it.")


def cmd_evaluate(args):
    with open(args.testset, encoding="utf-8") as f:
        testset = json.load(f)
    positives = [t for t in testset if t.get("expected")]
    negatives = [t for t in testset if not t.get("expected")]
    print(f"{len(positives)} positive / {len(negatives)} negative questions, K={K}, "
          f"coverage check={'on' if args.with_coverage else 'off'}\n")

    print("-" * 78)
    print(f"{'threshold':>9} | {'Recall@K':>8} | {'Precision@K':>11} | {'MRR':>3} | "
          f"{'avg kept':>8} | {'negatives rejected':>18}")
    print("-" * 78)
    for threshold in (args.thresholds or SWEEP):
        rows = []
        for item in positives:
            _, sources, _ = retrieval.run_retrieval_pipeline(
                item["question"], threshold=threshold,
                use_coverage_check=args.with_coverage, top_k=K,
            )
            rows.append(score_query(sources, item["expected"], K))
        rejected = 0
        for item in negatives:
            docs, _, _ = retrieval.run_retrieval_pipeline(
                item["question"], threshold=threshold,
                use_coverage_check=args.with_coverage, top_k=K,
            )
            rejected += int(not docs)

        def avg(key):
            return statistics.mean(r[key] for r in rows) if rows else 0.0

        neg = f"{rejected}/{len(negatives)}" if negatives else "n/a"
        print(f"{threshold:>9.2f} | {avg('recall'):>8.2f} | {avg('precision'):>11.2f} | "
              f"{avg('rr'):>5.2f} | {avg('returned'):>8.1f} | {neg:>18}")

    print("\nChoose the highest threshold where Recall@K has NOT dropped.")
    print("For medical evidence, losing recall (missing a real source) is worse than")
    print("lower precision, because the coverage check and validator filter noise later.")
    print("Set it in .env:  RAG_RELEVANCE_THRESHOLD=<value>")

def cmd_template(args):
    template = [
        {"question": "How is dengue transmitted?",
         "expected": [{"source": "REPLACE_WITH_YOUR.pdf", "page": 1}]},
        {"question": "What are the warning signs of severe dengue?",
         "expected": [{"source": "REPLACE_WITH_YOUR.pdf"}]},
        {"question": "What is the capital of France?", "expected": []},
    ]
    with open("rag_testset.json", "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print("Wrote rag_testset.json — replace the examples with 20-50 real questions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--queries", nargs="*")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--testset", default="rag_testset.json")
    p_eval.add_argument("--thresholds", nargs="*", type=float)
    p_eval.add_argument("--with-coverage", action="store_true")
    sub.add_parser("template")
    args = parser.parse_args()
    {"inspect": cmd_inspect, "evaluate": cmd_evaluate, "template": cmd_template}[args.cmd](args)
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
  {"question": "What is the capital of France?", "expected": []},   <- negative: should return nothing

  # Optional "disease" field: when present on a positive question, `evaluate`
  # ALSO runs that question through retrieval's disease filter (the same
  # disease=... argument retrieve_node passes in production) and prints a
  # separate "Disease filter impact" table comparing filtered vs. unfiltered
  # retrieval for Recall@K, Precision@K, and cross-disease chunk leakage.
  # Use one of config.KNOWN_DISEASES (malaria, dengue, chikungunya, zika).
  {"question": "What are the warning signs of severe dengue?",
   "expected": [{"source": "dengue_guidelines.pdf"}], "disease": "dengue"}
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


def count_cross_disease_leakage(sources: list, disease: str, k: int) -> int:
    """
    Count top-K sources tagged with a DIFFERENT disease than the one this
    question is about. "general" and untagged sources are never leakage —
    they're the shared/legacy documents retrieval is meant to let through
    regardless of which disease filter is active.
    """
    leaked = 0
    for source in sources[:k]:
        tag = source.get("disease")
        if tag and tag not in {disease, "general", "unspecified"}:
            leaked += 1
    return leaked


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

    # Single-threshold comparison of retrieve_node's actual disease filter,
    # only for questions tagged with a "disease" (see module docstring).
    impact_threshold = args.disease_threshold if args.disease_threshold is not None else retrieval.RELEVANCE_THRESHOLD
    report_disease_filter_impact(testset, impact_threshold, args.with_coverage, K)


def report_disease_filter_impact(testset: list, threshold: float, with_coverage: bool, k: int):
    """
    For every positive question tagged with a "disease", run retrieval BOTH
    unfiltered and filtered (disease=...), and compare: does the filter
    improve Recall@K / Precision@K, and does it actually stop cross-disease
    chunks from showing up? This is the direct evidence that retrieve_node's
    disease filter (added for the vector-borne-disease expansion) earns its
    keep, rather than just assuming a metadata filter helps.
    """
    tagged = [t for t in testset if t.get("expected") and t.get("disease")]
    if not tagged:
        print("\n(No test-set questions have a \"disease\" field — skipping the "
              "disease-filter-impact report. Add one to see filtered-vs-unfiltered results.)")
        return

    print(f"\nDisease filter impact (threshold={threshold:.2f}, "
          f"coverage={'on' if with_coverage else 'off'}, K={k})")
    print("-" * 96)
    header = (f"{'disease':>12} | {'n':>2} | {'recall':>7} {'→':>1} {'recall':>7} | "
              f"{'precision':>9} {'→':>1} {'precision':>9} | "
              f"{'leak/q':>7} {'→':>1} {'leak/q':>7} | {'fallback':>9}")
    print(header + "\n" + " " * 16 + "(no filter → filtered)" + " " * 6 +
          "(no filter → filtered)" + "    (no filter → filtered)")
    print("-" * 96)

    by_disease: dict = {}
    for item in tagged:
        disease = item["disease"]
        unfiltered_docs, unfiltered_sources, _ = retrieval.run_retrieval_pipeline(
            item["question"], threshold=threshold, use_coverage_check=with_coverage,
            top_k=k, disease=None,
        )
        filtered_docs, filtered_sources, filtered_diag = retrieval.run_retrieval_pipeline(
            item["question"], threshold=threshold, use_coverage_check=with_coverage,
            top_k=k, disease=disease,
        )
        row = {
            "unfiltered": score_query(unfiltered_sources, item["expected"], k),
            "filtered": score_query(filtered_sources, item["expected"], k),
            "leak_unfiltered": count_cross_disease_leakage(unfiltered_sources, disease, k),
            "leak_filtered": count_cross_disease_leakage(filtered_sources, disease, k),
            "fellback": filtered_diag.get("disease_filter_fellback", False),
        }
        by_disease.setdefault(disease, []).append(row) ##############

    def avg(rows, path):
        keys = path.split(".")
        vals = []
        for r in rows:
            v = r
            for key in keys:
                v = v[key]
            vals.append(v)
        return statistics.mean(vals) if vals else 0.0

    all_rows = []
    for disease in sorted(by_disease):
        rows = by_disease[disease]
        all_rows.extend(rows)
        fellback_count = sum(1 for r in rows if r["fellback"])
        print(f"{disease:>12} | {len(rows):>2} | "
              f"{avg(rows,'unfiltered.recall'):>7.2f} → {avg(rows,'filtered.recall'):>7.2f} | "
              f"{avg(rows,'unfiltered.precision'):>9.2f} → {avg(rows,'filtered.precision'):>9.2f} | "
              f"{avg(rows,'leak_unfiltered'):>7.1f} → {avg(rows,'leak_filtered'):>7.1f} | "
              f"{fellback_count}/{len(rows):>7}")

    fellback_total = sum(1 for r in all_rows if r["fellback"])
    print("-" * 96)
    print(f"{'TOTAL':>12} | {len(all_rows):>2} | "
          f"{avg(all_rows,'unfiltered.recall'):>7.2f} → {avg(all_rows,'filtered.recall'):>7.2f} | "
          f"{avg(all_rows,'unfiltered.precision'):>9.2f} → {avg(all_rows,'filtered.precision'):>9.2f} | "
          f"{avg(all_rows,'leak_unfiltered'):>7.1f} → {avg(all_rows,'leak_filtered'):>7.1f} | "
          f"{fellback_total}/{len(all_rows):>7}")

    print("\nRead this as: recall should not drop filtered vs. unfiltered (the filter should")
    print("never hide the right answer), while leak/q should drop toward 0 (the filter should")
    print("stop other diseases' chunks from entering the evidence). \"fallback\" counts how many")
    print("questions found nothing under their own disease + general and fell back to an")
    print("unfiltered search — a high fallback count usually means that disease's folder is")
    print("thin or mis-tagged, not that the filter logic is wrong.")


def cmd_template(args):
    template = [

        {"question": "How is dengue transmitted?",
         "expected": [{"source": "REPLACE_WITH_YOUR_DENGUE.pdf", "page": 1}]},
        {"question": "What is the capital of France?", "expected": []},

        {"question": "What are the warning signs of severe dengue?",
         "expected": [{"source": "REPLACE_WITH_YOUR_DENGUE.pdf"}], "disease": "dengue"},
        {"question": "What are the symptoms of malaria?",
         "expected": [{"source": "REPLACE_WITH_YOUR_MALARIA.pdf"}], "disease": "malaria"},
        {"question": "What joint symptoms does chikungunya cause?",
         "expected": [{"source": "REPLACE_WITH_YOUR_CHIKUNGUNYA.pdf"}], "disease": "chikungunya"},
        {"question": "How does Zika virus affect pregnancy?",
         "expected": [{"source": "REPLACE_WITH_YOUR_ZIKA.pdf"}], "disease": "zika"},
    ]
    with open("rag_testset.json", "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print("Wrote rag_testset.json — replace the examples with 20-50 real questions.")
    print("Tag positives with \"disease\" (malaria/dengue/chikungunya/zika) to also get")
    print("the disease-filter-impact report when you run `evaluate`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--queries", nargs="*")
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--testset", default="rag_testset.json")
    p_eval.add_argument("--thresholds", nargs="*", type=float)
    p_eval.add_argument("--with-coverage", action="store_true")
    p_eval.add_argument(
        "--disease-threshold", type=float, default=None,
        help="Threshold used for the disease-filter-impact report "
             "(default: retrieval.RELEVANCE_THRESHOLD, i.e. what production uses).",
    )
    sub.add_parser("template")
    args = parser.parse_args()
    {"inspect": cmd_inspect, "evaluate": cmd_evaluate, "template": cmd_template}[args.cmd](args)
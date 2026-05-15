#!/usr/bin/env python3
"""Rank gathered papers by applicability to the RDLM/ARC codebase.

Input is the JSONL produced by scripts/gather_research.py. Output is a markdown
bibliography with a compact entry for every paper and deeper synthesis prompts
for the top N. The scoring function is intentionally transparent and keyword
based so future agents can adjust it without model/API dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DIRECT_CODE_TERMS = {
    "arc": 8,
    "arc-agi": 10,
    "abstraction and reasoning corpus": 10,
    "sudoku": 5,
    "maze": 5,
    "program synthesis": 6,
    "masked diffusion": 8,
    "diffusion language": 8,
    "discrete diffusion": 6,
    "recursive": 7,
    "recurrent": 4,
    "iterative refinement": 6,
    "test-time": 6,
    "test time": 6,
    "verifier": 5,
    "self-consistency": 5,
    "small language model": 6,
    "distillation": 4,
    "object-centric": 6,
    "neuro-symbolic": 5,
    "state space": 3,
    "mamba": 3,
}

THEME_TERMS: dict[str, tuple[str, ...]] = {
    "Recursive / iterative reasoning": ("recursive", "recurrent", "iterative", "refinement", "adaptive computation"),
    "Small-model reasoning": ("small language", "tiny", "efficient", "distillation", "sample efficiency"),
    "Diffusion / denoising LM": ("diffusion", "denoising", "masked", "discrete diffusion"),
    "Test-time compute / verification": ("test-time", "test time", "verifier", "self-consistency", "tree search", "rerank", "self correction"),
    "ARC / program / symbolic reasoning": ("arc", "abstraction and reasoning", "program synthesis", "neuro-symbolic", "symbolic", "sudoku", "maze"),
    "Memory / efficient sequence models": ("memory", "state space", "mamba", "rwkv", "sparse attention", "mixture of experts"),
    "Curriculum / synthetic data / RL": ("curriculum", "synthetic", "reinforcement learning", "rl", "data generation"),
    "Mechanistic interpretability": ("mechanistic", "interpretability", "induction", "circuits"),
}

PENALTY_TERMS = {
    "billion parameters": -4,
    "trillion": -5,
    "large language model": -1,
    "100b": -4,
    "70b": -3,
}

NOISY_DOMAIN_TERMS = (
    "medical",
    "medicine",
    "clinical",
    "biomedical",
    "healthcare",
    "patient",
    "med-palm",
    "materials discovery",
    "materials science",
    "molecular",
    "drug discovery",
    "chemistry",
    "crystal",
    "protein",
)


def text_of(paper: dict[str, Any]) -> str:
    return "\n".join(
        str(paper.get(k) or "")
        for k in ("title", "abstract", "venue", "query_group", "source_query")
    ).lower()


def has_term(text: str, term: str) -> bool:
    """Match terms without letting short tokens like ARC fire inside words."""
    if re.fullmatch(r"[a-z0-9-]+", term) and len(term) <= 4:
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def score_paper(paper: dict[str, Any]) -> tuple[float, dict[str, float]]:
    text = text_of(paper)
    parts: dict[str, float] = {}
    parts["direct_code_relevance"] = sum(weight for term, weight in DIRECT_CODE_TERMS.items() if has_term(text, term))
    parts["topic_coverage"] = min(10.0, 1.5 * len(paper.get("topics") or []))
    year = paper.get("year") or 0
    parts["recency"] = max(0.0, min(6.0, (year - 2019) * 0.8)) if isinstance(year, int) else 0.0
    citations = paper.get("citation_count") or 0
    parts["impact"] = min(8.0, math.log10(citations + 1) * 3.0)
    parts["small_model_bonus"] = 3.0 if any(has_term(text, t) for t in ("small", "tiny", "efficient", "distillation", "sample efficient")) else 0.0
    parts["reasoning_bonus"] = 4.0 if "reason" in text or "planning" in text or "proof" in text else 0.0
    parts["cost_penalty"] = sum(weight for term, weight in PENALTY_TERMS.items() if has_term(text, term))
    score = sum(parts.values())
    return score, parts


def infer_theme(paper: dict[str, Any]) -> str:
    text = text_of(paper)
    hits = {
        theme: sum(1 for term in terms if has_term(text, term))
        for theme, terms in THEME_TERMS.items()
    }
    best, count = max(hits.items(), key=lambda kv: kv[1])
    return best if count else "Other relevant ML/AI"


def compact_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown authors"
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def clean_line(text: str | None, limit: int = 280) -> str:
    if not text:
        return "No abstract available."
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def implementation_relevance(paper: dict[str, Any], theme: str) -> str:
    title_abs = text_of(paper)
    if has_term(title_abs, "arc") or "arc-agi" in title_abs or "abstraction and reasoning" in title_abs:
        return "Directly relevant to ARC evaluation, object/grid priors, or candidate verification."
    if "diffusion" in title_abs or "masked" in title_abs or "denoising" in title_abs:
        return "Relevant to RDLM masking schedules, denoising objectives, and decoding trajectories."
    if "recursive" in title_abs or "recurrent" in title_abs or "iterative" in title_abs:
        return "Relevant to TRM-style recurrent refinement and adaptive computation depth."
    if "verifier" in title_abs or "self-consistency" in title_abs or "test-time" in title_abs or "test time" in title_abs:
        return "Relevant to low-risk inference-time reranking, voting, and extra-compute allocation."
    if "small" in title_abs or "distillation" in title_abs or "efficient" in title_abs:
        return "Relevant to keeping model size/training cost low while improving reasoning quality."
    if "program" in title_abs or "symbolic" in title_abs:
        return "Relevant to neuro-symbolic or program-like biases for abstract reasoning tasks."
    return f"Potentially relevant under theme: {theme}."


def load_papers(path: Path) -> list[dict[str, Any]]:
    papers = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return papers


def is_noisy_domain(paper: dict[str, Any]) -> bool:
    text = text_of(paper)
    return any(has_term(text, term) for term in NOISY_DOMAIN_TERMS)


def write_markdown(
    papers: list[dict[str, Any]],
    out: Path,
    deep_top_n: int,
    exclude_noisy_domains: bool,
) -> None:
    original_count = len(papers)
    if exclude_noisy_domains:
        papers = [paper for paper in papers if not is_noisy_domain(paper)]
    filtered_count = original_count - len(papers)

    scored = []
    for paper in papers:
        score, parts = score_paper(paper)
        theme = infer_theme(paper)
        paper = dict(paper)
        paper["applicability_score"] = round(score, 2)
        paper["score_parts"] = parts
        paper["theme"] = theme
        scored.append(paper)
    scored.sort(key=lambda p: p["applicability_score"], reverse=True)

    theme_counts = Counter(p["theme"] for p in scored)
    by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in scored:
        by_theme[paper["theme"]].append(paper)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Ranked Research Bibliography for RDLM\n\n")
        f.write(f"Total gathered papers after filtering: **{len(scored)}**.\n")
        if exclude_noisy_domains:
            f.write(f"Filtered noisy medicine/materials/domain-specific papers: **{filtered_count}**.\n")
        f.write("\n")
        f.write("## Ranking formula\n\n")
        f.write(
            "Keyword score over title/abstract/query metadata: direct code relevance + topic coverage "
            "+ recency + citation impact + small-model/reasoning bonuses - large-scale penalties.\n\n"
        )
        f.write("## Theme counts\n\n")
        for theme, count in theme_counts.most_common():
            f.write(f"- **{theme}**: {count}\n")
        f.write("\n## Top papers for deep synthesis\n\n")
        for idx, paper in enumerate(scored[:deep_top_n], start=1):
            f.write(f"### {idx}. {paper['title']}\n")
            f.write(f"- **Score**: {paper['applicability_score']}\n")
            f.write(f"- **Theme**: {paper['theme']}\n")
            f.write(f"- **Authors**: {compact_authors(paper.get('authors') or [])}\n")
            f.write(f"- **Year / venue**: {paper.get('year') or 'n/a'} / {paper.get('venue') or 'n/a'}\n")
            f.write(f"- **URL**: {paper.get('url') or 'n/a'}\n")
            f.write(f"- **Topics**: {', '.join(paper.get('topics') or [])}\n")
            f.write(f"- **Factual summary seed**: {clean_line(paper.get('abstract'))}\n")
            f.write(f"- **Implementation relevance**: {implementation_relevance(paper, paper['theme'])}\n")
            f.write("- **Deep-synthesis fields to fill**: method; architecture; training objective; benchmark; result; implementation hook; risks.\n\n")

        f.write("## Categorized bibliography\n\n")
        for theme, papers_for_theme in sorted(by_theme.items()):
            f.write(f"### {theme}\n\n")
            for paper in papers_for_theme:
                year = paper.get("year") or "n/a"
                url = paper.get("url") or ""
                citations = paper.get("citation_count")
                cite_text = f", cites={citations}" if citations is not None else ""
                f.write(
                    f"- **{paper['title']}** ({year}, score={paper['applicability_score']}{cite_text})"
                    f" — {compact_authors(paper.get('authors') or [])}. {url}\n"
                )
            f.write("\n")

    json_out = out.with_suffix(".scored.jsonl")
    with json_out.open("w", encoding="utf-8") as f:
        for paper in scored:
            f.write(json.dumps(paper, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote markdown to {out}")
    print(f"wrote scored JSONL to {json_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/research/raw_papers.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/research/ranked_papers.md"))
    parser.add_argument("--deep-top-n", type=int, default=75)
    parser.add_argument(
        "--include-noisy-domains",
        action="store_true",
        help="Keep medicine/materials/chemistry papers instead of filtering them from ranked outputs.",
    )
    args = parser.parse_args()
    papers = load_papers(args.input)
    write_markdown(
        papers,
        args.out,
        args.deep_top_n,
        exclude_noisy_domains=not args.include_noisy_domains,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

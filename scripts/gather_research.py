#!/usr/bin/env python3
"""Gather candidate ML/AI research papers from public metadata APIs.

The script intentionally uses only Python's standard library so it can run in
this repository without dependency changes. It queries Semantic Scholar, arXiv,
and OpenAlex with focused search terms, excludes papers already mentioned in an
existing review file, deduplicates records, and writes JSONL metadata suitable
for later ranking/synthesis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

USER_AGENT = "rdlm-research-gatherer/1.0 (mailto:research@example.invalid)"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

QUERY_GROUPS: dict[str, list[str]] = {
    "recursive_reasoning": [
        '"recursive reasoning" neural network',
        '"hierarchical reasoning model" neural reasoning',
        '"iterative refinement" transformer reasoning',
        '"recurrent" "reasoning" "transformer"',
        '"adaptive computation" neural reasoning',
    ],
    "small_language_models": [
        '"small language model" reasoning',
        '"small language models" "chain of thought"',
        '"language model" "sample efficiency" reasoning',
        '"distillation" "reasoning" "small language model"',
        '"efficient transformers" reasoning language models',
    ],
    "diffusion_text": [
        '"masked diffusion" language model',
        '"discrete diffusion" text generation',
        '"diffusion language model" reasoning',
        '"non autoregressive" language model iterative refinement',
        '"denoising" "language model" "reasoning"',
    ],
    "test_time_compute": [
        '"test-time compute" reasoning language models',
        '"test time scaling" reasoning',
        '"self-consistency" reasoning language model',
        '"verifier guided" language model reasoning',
        '"self correction" language model reasoning',
        '"tree search" language model reasoning',
    ],
    "latent_cot_planning": [
        '"latent reasoning" language model',
        '"chain of thought" "small language model"',
        '"planning" "language model" "reasoning"',
        '"process supervision" reasoning language model',
        '"neural theorem proving" language model reasoning',
    ],
    "arc_abstract_reasoning": [
        '"ARC-AGI" neural symbolic',
        '"Abstraction and Reasoning Corpus" neural',
        '"ARC" "program synthesis" reasoning',
        '"Sudoku" "neural reasoning"',
        '"maze" "neural reasoning"',
        '"abstract reasoning" neural network benchmark',
    ],
    "neuro_symbolic_programs": [
        '"neuro-symbolic" reasoning neural network',
        '"program synthesis" "language model" reasoning',
        '"inductive logic programming" neural reasoning',
        '"differentiable" "program synthesis" reasoning',
    ],
    "memory_state_space_sparse": [
        '"memory augmented" transformer reasoning',
        '"Mamba" language model reasoning',
        '"state space model" language model reasoning',
        '"RWKV" language model reasoning',
        '"sparse attention" reasoning language model',
        '"mixture of experts" reasoning small language model',
    ],
    "curriculum_synthetic_rl": [
        '"curriculum learning" abstract reasoning',
        '"synthetic data" reasoning small language model',
        '"reinforcement learning" reasoning language model verifier',
        '"RL" "chain of thought" reasoning',
        '"data generation" "reasoning" "language model"',
    ],
    "mechanistic_interpretability": [
        '"mechanistic interpretability" reasoning language models',
        '"induction heads" reasoning transformer',
        '"circuits" "reasoning" "transformer"',
    ],
}

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "recursive reasoning": ("recursive", "recursion", "recurrent", "iterative refinement", "hierarchical reasoning"),
    "small models": ("small language", "tiny", "efficient", "distillation", "parameter-efficient"),
    "diffusion LM": ("diffusion", "masked", "denoising", "non-autoregressive", "non autoregressive"),
    "test-time compute": ("test-time", "test time", "self-consistency", "tree search", "verifier", "rerank"),
    "latent/CoT reasoning": ("chain of thought", "cot", "latent reasoning", "process supervision", "reasoning trace"),
    "ARC/abstract reasoning": ("arc-agi", "abstraction and reasoning", "sudoku", "maze", "abstract reasoning"),
    "neuro-symbolic/programs": ("neuro-symbolic", "neurosymbolic", "program synthesis", "logic", "symbolic"),
    "memory/state-space/sparse": ("memory", "mamba", "state space", "rwkv", "sparse attention", "mixture of experts", "moe"),
    "curriculum/synthetic/RL": ("curriculum", "synthetic", "reinforcement learning", "rl", "data generation"),
    "mechanistic interpretability": ("mechanistic", "interpretability", "induction head", "circuit"),
}


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    semantic_scholar_id: str | None = None
    openalex_id: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    source: str = ""
    source_query: str = ""
    query_group: str = ""
    topics: list[str] = field(default_factory=list)


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def extract_arxiv_id(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(?:arxiv.org/(?:abs|pdf)/|arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", text, re.I)
    return match.group(1) if match else None


def request_json(url: str, *, timeout: int = 30, retries: int = 2) -> Any | None:
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                print(f"warning: failed JSON request {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def request_text(url: str, *, timeout: int = 30, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                print(f"warning: failed text request {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def infer_topics(title: str, abstract: str | None, query_group: str) -> list[str]:
    text = f"{title}\n{abstract or ''}".lower()
    topics = [topic for topic, words in TOPIC_KEYWORDS.items() if any(word in text for word in words)]
    group_topic = query_group.replace("_", " ")
    if group_topic not in topics:
        topics.append(group_topic)
    return topics


def semantic_scholar_search(query: str, group: str, limit: int) -> Iterable[Paper]:
    fields = ",".join([
        "title",
        "authors",
        "year",
        "venue",
        "url",
        "abstract",
        "citationCount",
        "externalIds",
    ])
    params = urllib.parse.urlencode({"query": query, "limit": limit, "fields": fields})
    data = request_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}")
    for item in (data or {}).get("data", []):
        title = (item.get("title") or "").strip()
        if not title:
            continue
        external = item.get("externalIds") or {}
        arxiv_id = external.get("ArXiv") or extract_arxiv_id(item.get("url"))
        abstract = item.get("abstract")
        yield Paper(
            title=title,
            authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
            year=item.get("year"),
            venue=item.get("venue"),
            url=item.get("url"),
            arxiv_id=arxiv_id,
            doi=external.get("DOI"),
            semantic_scholar_id=item.get("paperId"),
            abstract=abstract,
            citation_count=item.get("citationCount"),
            source="semantic_scholar",
            source_query=query,
            query_group=group,
            topics=infer_topics(title, abstract, group),
        )


def arxiv_search(query: str, group: str, limit: int) -> Iterable[Paper]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    text = request_text(f"http://export.arxiv.org/api/query?{params}")
    if not text:
        return
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        print(f"warning: failed to parse arXiv response for {query}: {exc}", file=sys.stderr)
        return
    for entry in root.findall("atom:entry", ARXIV_NS):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ARXIV_NS)).split())
        if not title:
            continue
        url = entry.findtext("atom:id", default=None, namespaces=ARXIV_NS)
        abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=ARXIV_NS)).split())
        published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            a.findtext("atom:name", default="", namespaces=ARXIV_NS)
            for a in entry.findall("atom:author", ARXIV_NS)
        ]
        yield Paper(
            title=title,
            authors=[a for a in authors if a],
            year=year,
            venue="arXiv",
            url=url,
            arxiv_id=extract_arxiv_id(url),
            abstract=abstract,
            source="arxiv",
            source_query=query,
            query_group=group,
            topics=infer_topics(title, abstract, group),
        )


def openalex_search(query: str, group: str, limit: int, mailto: str | None) -> Iterable[Paper]:
    params = {"search": query, "per-page": limit, "sort": "relevance_score:desc"}
    if mailto:
        params["mailto"] = mailto
    data = request_json(f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}")
    for item in (data or {}).get("results", []):
        title = (item.get("title") or item.get("display_name") or "").strip()
        if not title:
            continue
        authorships = item.get("authorships") or []
        authors = []
        for authorship in authorships:
            author = authorship.get("author") or {}
            if author.get("display_name"):
                authors.append(author["display_name"])
        abstract = openalex_abstract(item.get("abstract_inverted_index"))
        ids = item.get("ids") or {}
        doi = item.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi.removeprefix("https://doi.org/")
        yield Paper(
            title=title,
            authors=authors,
            year=item.get("publication_year"),
            venue=((item.get("primary_location") or {}).get("source") or {}).get("display_name"),
            url=ids.get("openalex") or item.get("id"),
            arxiv_id=extract_arxiv_id(json.dumps(ids)),
            doi=doi,
            openalex_id=item.get("id"),
            abstract=abstract,
            citation_count=item.get("cited_by_count"),
            source="openalex",
            source_query=query,
            query_group=group,
            topics=infer_topics(title, abstract, group),
        )


def openalex_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        words.extend((pos, word) for pos in positions)
    return " ".join(word for _pos, word in sorted(words))


def load_exclusions(path: Path | None) -> tuple[set[str], set[str], set[str]]:
    titles: set[str] = set()
    arxiv_ids: set[str] = set()
    dois: set[str] = set()
    if path is None or not path.exists():
        return titles, arxiv_ids, dois
    text = path.read_text(encoding="utf-8")
    titles.update(normalize_title(m.group(1)) for m in re.finditer(r"^###\s+(.+)$", text, re.M))
    titles.update(normalize_title(m.group(1)) for m in re.finditer(r"^\|\s*\d{4}\s*\|\s*([^|]+?)\s*\|", text, re.M))
    arxiv_ids.update(m.group(1) for m in re.finditer(r"arxiv(?:\.org/abs/|\s+)(\d{4}\.\d{4,5})", text, re.I))
    dois.update(m.group(1).lower() for m in re.finditer(r"doi\.org/([^\s)]+)", text, re.I))
    return titles, arxiv_ids, dois


def paper_key(paper: Paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.lower()}"
    if paper.semantic_scholar_id:
        return f"s2:{paper.semantic_scholar_id}"
    if paper.openalex_id:
        return f"oa:{paper.openalex_id}"
    return f"title:{normalize_title(paper.title)}"


def merge_papers(existing: Paper, incoming: Paper) -> Paper:
    merged = Paper(**asdict(existing))
    for field_name in ("year", "venue", "url", "arxiv_id", "doi", "semantic_scholar_id", "openalex_id", "abstract", "citation_count"):
        if getattr(merged, field_name) in (None, "") and getattr(incoming, field_name) not in (None, ""):
            setattr(merged, field_name, getattr(incoming, field_name))
    if incoming.citation_count is not None and (merged.citation_count is None or incoming.citation_count > merged.citation_count):
        merged.citation_count = incoming.citation_count
    merged.authors = list(dict.fromkeys([*merged.authors, *incoming.authors]))
    merged.topics = sorted(set([*merged.topics, *incoming.topics]))
    sources = set(filter(None, merged.source.split("+"))) | {incoming.source}
    merged.source = "+".join(sorted(sources))
    if incoming.source_query not in merged.source_query:
        merged.source_query = "; ".join(filter(None, [merged.source_query, incoming.source_query]))
    return merged


def is_excluded(paper: Paper, excluded_titles: set[str], excluded_arxiv: set[str], excluded_dois: set[str]) -> bool:
    title = normalize_title(paper.title)
    return (
        title in excluded_titles
        or bool(paper.arxiv_id and paper.arxiv_id in excluded_arxiv)
        or bool(paper.doi and paper.doi.lower() in excluded_dois)
    )


def iter_queries(selected_groups: list[str] | None) -> Iterable[tuple[str, str]]:
    groups = selected_groups or list(QUERY_GROUPS)
    for group in groups:
        if group not in QUERY_GROUPS:
            raise SystemExit(f"unknown query group {group!r}; choices: {', '.join(QUERY_GROUPS)}")
        for query in QUERY_GROUPS[group]:
            yield group, query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-papers", type=int, default=200, help="Target number of deduped papers to gather.")
    parser.add_argument("--per-query", type=int, default=20, help="Results requested per API per query.")
    parser.add_argument("--exclude", type=Path, default=Path("research_findings.md"), help="Markdown review whose papers should be excluded.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/research/raw_papers.jsonl"))
    parser.add_argument("--groups", nargs="*", help="Optional subset of query groups to run.")
    parser.add_argument("--sources", nargs="*", default=["semantic_scholar", "arxiv", "openalex"], choices=["semantic_scholar", "arxiv", "openalex"], help="Metadata APIs to query.")
    parser.add_argument("--mailto", default=None, help="Optional email for OpenAlex polite pool.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between API requests.")
    args = parser.parse_args()

    excluded_titles, excluded_arxiv, excluded_dois = load_exclusions(args.exclude)
    papers: dict[str, Paper] = {}
    seen_title_to_key: dict[str, str] = {}

    for group, query in iter_queries(args.groups):
        print(f"query [{group}] {query}", file=sys.stderr)
        providers = []
        if "semantic_scholar" in args.sources:
            providers.append(lambda q=query, g=group: semantic_scholar_search(q, g, args.per_query))
        if "arxiv" in args.sources:
            providers.append(lambda q=query, g=group: arxiv_search(q, g, args.per_query))
        if "openalex" in args.sources:
            providers.append(lambda q=query, g=group: openalex_search(q, g, args.per_query, args.mailto))
        for provider in providers:
            for paper in provider() or []:
                if is_excluded(paper, excluded_titles, excluded_arxiv, excluded_dois):
                    continue
                key = paper_key(paper)
                title_key = normalize_title(paper.title)
                if title_key in seen_title_to_key:
                    key = seen_title_to_key[title_key]
                else:
                    seen_title_to_key[title_key] = key
                papers[key] = merge_papers(papers[key], paper) if key in papers else paper
            time.sleep(args.sleep)
        print(f"deduped papers so far: {len(papers)}", file=sys.stderr)
        if len(papers) >= args.min_papers:
            # Continue current breadth only if groups explicitly selected? Stop by default to be polite.
            break

    ordered = sorted(
        papers.values(),
        key=lambda p: ((p.citation_count or 0), p.year or 0, p.title),
        reverse=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for paper in ordered:
            f.write(json.dumps(asdict(paper), ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(ordered)} papers to {args.out}")
    if len(ordered) < args.min_papers:
        print(f"warning: target was {args.min_papers}; try increasing --per-query or adding groups", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

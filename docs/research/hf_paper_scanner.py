#!/usr/bin/env python3
"""
ModelCypher Daily Paper Scanner
================================
Pulls yesterday's HuggingFace Daily Papers, filters for geometric/algebraic/
mechanistic relevance to ModelCypher, and produces a markdown report.

Usage:
    python hf_paper_scanner.py                    # yesterday's papers
    python hf_paper_scanner.py --date 2026-02-24  # specific date
    python hf_paper_scanner.py --days 3           # last 3 days

Output: research/paper_reviews/YYYY-MM-DD_geometric_review.md
"""

import json
import re
import sys
import os
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────

HF_API = "https://huggingface.co/api/daily_papers"
ARXIV_API = "https://export.arxiv.org/api/query"
OUTPUT_DIR = Path(__file__).parent / "paper_reviews"

# ─── Geometric relevance keywords (weighted) ────────────────────────────────

# Tier 1: Direct geometric/spectral terms (high signal)
TIER1_KEYWORDS = [
    r"\bsvd\b", r"\bsingular value", r"\bspectral\b", r"\beigenvalue",
    r"\beigenvector", r"\blow.rank", r"\brank\b.*\b(matrix|weight|adapter|approximat)",
    r"\briemannian\b", r"\bstiefel\b", r"\bgrassmann", r"\bcayley\b",
    r"\bmanifold\b", r"\bgeodesic\b", r"\bcurvature\b", r"\btopolog",
    r"\bpersistent homology", r"\bbetti\b", r"\bsimplicial",
    r"\bprocrustes\b", r"\bcka\b", r"\bcentered kernel alignment",
    r"\bgromov.wasserstein", r"\boptimal transport",
    r"\bintrinsic dimension", r"\beffective rank", r"\bstable rank",
    r"\bnull.space", r"\borthogonal\b.*\b(constraint|parameteriz|project)",
    r"\bnorm.bound", r"\bspectral norm", r"\bfrobenius",
    r"\bweyl\b", r"\bperturbation bound",
    r"\blora\b.*\b(spectral|rank|svd|geometric|norm)",
    r"\bweight matrix\b.*\b(geometry|spectral|svd|decompos)",
]

# Tier 2: Mechanistic/structural terms (medium signal)
TIER2_KEYWORDS = [
    r"\bmechanistic interpret", r"\bcircuit\b.*\b(neural|transformer|attention)",
    r"\brepresentation\b.*\b(geometry|structure|space|manifold|alignment)",
    r"\bactivation\b.*\b(manifold|geometry|space|spectral|svd)",
    r"\blayer.wise\b.*\b(analysis|geometry|spectral|rank|dimension)",
    r"\btransformer\b.*\b(geometry|spectral|composition|operator)",
    r"\bcomposition\b.*\b(transform|operator|function|layer)",
    r"\binformation geometry", r"\bfisher\b.*\b(information|metric|matrix)",
    r"\bnatural gradient", r"\bprecondition",
    r"\bconvergence\b.*\b(riemannian|spectral|geometric|manifold)",
    r"\blearning rate\b.*\b(spectral|adaptive|geometric|bound|deriv)",
    r"\bweight decay\b.*\b(spectral|geometric|condition)",
    r"\bgradient\b.*\b(spectral|geometric|noise|structure)",
    r"\brepresentation\b.*\b(similar|compari|align|converg)",
    r"\bplatonic\b", r"\buniversal\b.*\brepresentation",
    r"\bbottleneck\b.*\b(dimension|layer|representation)",
    r"\babstraction\b.*\b(phase|layer|dimension)",
    r"\bdeterministic\b.*\b(map|transform|forward)",
]

# Tier 3: Adjacent terms (low signal individually, but supportive)
TIER3_KEYWORDS = [
    r"\blora\b", r"\badapter\b", r"\bfine.tun",
    r"\bloss landscape", r"\boptimiz.*landscape",
    r"\bhessian\b", r"\bjacobian\b",
    r"\bkernel\b.*\b(method|trick|alignment)",
    r"\blinear algebra", r"\bmatrix\b.*\b(decompos|factor)",
    r"\bsubspace\b", r"\bprojection\b",
    r"\bembedding\b.*\b(space|geometry|structure)",
    r"\binterfer\b.*\b(task|catastrophic|knowledge)",
    r"\bmerge\b.*\b(model|weight|adapter)",
    r"\bpruning\b.*\b(spectral|structured|geometric)",
    r"\battention\b.*\b(rank|spectral|head|pattern)",
]

# Anti-keywords: reduce score if these dominate
ANTI_KEYWORDS = [
    r"\bbenchmark\b.*\b(sota|state.of.the.art|new record)",
    r"\bprompt\b.*\b(engineer|template|chain.of.thought)",
    r"\bscaling law\b(?!.*spectral)",
    r"\brag\b(?!.*geometric)", r"\bagent\b.*\b(framework|tool.use)",
    r"\breinforcement learning from human",
    r"\bdataset\b.*\b(new|curated|collected)",
    r"\bvision.language\b(?!.*geometric)",
    r"\bmultimodal\b(?!.*representation.*geometry)",
]

# Context: paper must be about language models / transformers / neural nets (not pure CV/3D/robotics)
LLM_CONTEXT_KEYWORDS = [
    r"\blanguage model", r"\bllm\b", r"\btransformer\b", r"\battention\b",
    r"\bneural network", r"\bdeep learning", r"\bweight\b.*\b(matrix|matric)",
    r"\bfine.tun", r"\bpre.train", r"\blora\b", r"\badapter\b",
    r"\btoken\b", r"\bembedding\b", r"\bmodel\b.*\b(train|weight|layer|param)",
    r"\bgradient\b", r"\bbackprop", r"\boptimiz",
    r"\binterpretab", r"\bmechanistic\b", r"\brepresentation\b.*\blearn",
    r"\bgeneraliz", r"\boverfit", r"\bconvergence\b",
    r"\bactivation\b", r"\bhidden\b.*\b(layer|state|represent)",
]


def score_paper(title: str, summary: str) -> tuple[float, list[str]]:
    """Score a paper for ModelCypher geometric relevance.

    Returns (score, list_of_matched_terms).
    Threshold for inclusion: score >= 3.0
    """
    text = f"{title} {summary}".lower()
    score = 0.0
    matches = []

    for pattern in TIER1_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            score += 3.0
            matches.append(f"T1:{pattern}")

    for pattern in TIER2_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1.5
            matches.append(f"T2:{pattern}")

    for pattern in TIER3_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.5
            matches.append(f"T3:{pattern}")

    # Anti-keyword penalty
    anti_count = 0
    for pattern in ANTI_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            anti_count += 1
    if anti_count > 0 and score < 6.0:
        score *= max(0.3, 1.0 - 0.25 * anti_count)

    # Context filter: if paper has NO LLM/neural-net context keywords AND
    # score is below high-signal threshold, it's likely a CV/robotics/3D paper
    # that matched on generic terms like "manifold" or "representation space"
    if score > 0 and score < 6.0:
        has_nn_context = any(
            re.search(p, text, re.IGNORECASE) for p in LLM_CONTEXT_KEYWORDS
        )
        if not has_nn_context:
            score *= 0.3  # heavy penalty — probably not about LLM internals
            matches.append("CONTEXT:no-llm-context-penalty")

    return score, matches


def fetch_daily_papers(date_str: str) -> list[dict]:
    """Fetch papers from HuggingFace Daily Papers API for a given date."""
    url = f"{HF_API}?date={date_str}&limit=100"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ModelCypher-Scanner/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data
    except urllib.error.URLError as e:
        print(f"[ERROR] Failed to fetch HF papers for {date_str}: {e}", file=sys.stderr)
        return []


def fetch_arxiv_abstract(arxiv_id: str) -> Optional[str]:
    """Fetch full abstract from arXiv API (HF summaries can be truncated)."""
    url = f"{ARXIV_API}?id_list={arxiv_id}&max_results=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ModelCypher-Scanner/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml = resp.read().decode()
        # Simple XML extraction — no external deps needed
        match = re.search(r"<summary>(.*?)</summary>", xml, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return None


def generate_report(date_str: str, papers: list[dict]) -> tuple[str, list[dict]]:
    """Generate the daily review markdown report. Returns (markdown, scored_papers)."""
    scored = []
    for entry in papers:
        paper = entry.get("paper", entry)
        title = paper.get("title", entry.get("title", ""))
        summary = paper.get("summary", entry.get("summary", ""))
        arxiv_id = paper.get("id", "")
        upvotes = paper.get("upvotes", entry.get("upvotes", 0))
        authors = paper.get("authors", [])
        github = paper.get("githubRepo", "")
        ai_keywords = paper.get("ai_keywords", [])

        score, matches = score_paper(title, summary)
        if score >= 3.0:
            scored.append({
                "title": title,
                "arxiv_id": arxiv_id,
                "summary": summary,
                "score": score,
                "matches": matches,
                "upvotes": upvotes,
                "authors": [a.get("name", "") for a in authors] if isinstance(authors, list) else [],
                "github": github,
                "ai_keywords": ai_keywords,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Build report
    lines = []
    lines.append(f"# ModelCypher Paper Review — {date_str}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Papers scanned**: {len(papers)}")
    lines.append(f"**Papers with geometric relevance**: {len(scored)}")
    lines.append("")

    if not scored:
        lines.append("No papers with geometric relevance found today.")
        lines.append("")
        lines.append("### Near-misses (score 1.5–3.0):")
        lines.append("")
        near_misses = []
        for entry in papers:
            paper = entry.get("paper", entry)
            title = paper.get("title", entry.get("title", ""))
            summary = paper.get("summary", entry.get("summary", ""))
            arxiv_id = paper.get("id", "")
            score, matches = score_paper(title, summary)
            if 1.5 <= score < 3.0:
                near_misses.append((title, arxiv_id, score, matches))
        near_misses.sort(key=lambda x: x[2], reverse=True)
        for title, aid, sc, mt in near_misses[:5]:
            lines.append(f"- **{title}** ([{aid}](https://arxiv.org/abs/{aid})) — score: {sc:.1f}")
        if not near_misses:
            lines.append("None.")
        lines.append("")
        return "\n".join(lines), []

    lines.append("---")
    lines.append("")

    for i, p in enumerate(scored, 1):
        arxiv_url = f"https://arxiv.org/abs/{p['arxiv_id']}"
        author_str = ", ".join(p["authors"][:5])
        if len(p["authors"]) > 5:
            author_str += f" et al. ({len(p['authors'])} total)"

        lines.append(f"## {i}. {p['title']}")
        lines.append("")
        lines.append(f"**arXiv**: [{p['arxiv_id']}]({arxiv_url})")
        lines.append(f"**Authors**: {author_str}")
        lines.append(f"**HF Upvotes**: {p['upvotes']}")
        lines.append(f"**Geometric Relevance Score**: {p['score']:.1f}")
        if p["github"]:
            lines.append(f"**Code**: [{p['github']}]({p['github']})")
        lines.append("")

        lines.append("### Summary")
        lines.append(p["summary"])
        lines.append("")

        lines.append("### Keyword Matches")
        # Group by tier
        t1 = [m.split(":", 1)[1] for m in p["matches"] if m.startswith("T1:")]
        t2 = [m.split(":", 1)[1] for m in p["matches"] if m.startswith("T2:")]
        t3 = [m.split(":", 1)[1] for m in p["matches"] if m.startswith("T3:")]
        if t1:
            lines.append(f"- **Core geometric** ({len(t1)}): {', '.join(t1[:5])}")
        if t2:
            lines.append(f"- **Mechanistic/structural** ({len(t2)}): {', '.join(t2[:5])}")
        if t3:
            lines.append(f"- **Adjacent** ({len(t3)}): {', '.join(t3[:5])}")
        lines.append("")

        lines.append("### ModelCypher Integration Notes")
        lines.append("<!-- FILL: Claude deep-dive pass populates this section -->")
        lines.append("_Pending deep analysis — run with --deep flag or review manually._")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary table
    lines.append("## Quick Reference")
    lines.append("")
    lines.append("| # | Score | Paper | arXiv | Code |")
    lines.append("|---|-------|-------|-------|------|")
    for i, p in enumerate(scored, 1):
        code_link = f"[repo]({p['github']})" if p['github'] else "—"
        short_title = p['title'][:60] + ("..." if len(p['title']) > 60 else "")
        lines.append(f"| {i} | {p['score']:.1f} | {short_title} | [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']}) | {code_link} |")
    lines.append("")

    return "\n".join(lines), scored


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ModelCypher HF Paper Scanner")
    parser.add_argument("--date", help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to scan (default: 1 = yesterday)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Minimum relevance score (default: 3.0)")
    parser.add_argument("--output", help="Output directory (default: research/paper_reviews/)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.date:
        dates = [args.date]
    else:
        dates = []
        for i in range(1, args.days + 1):
            d = datetime.now() - timedelta(days=i)
            dates.append(d.strftime("%Y-%m-%d"))

    for date_str in dates:
        print(f"[SCAN] Fetching papers for {date_str}...")
        papers = fetch_daily_papers(date_str)
        print(f"[SCAN] Found {len(papers)} papers")

        report, scored_papers = generate_report(date_str, papers)

        outfile = output_dir / f"{date_str}_geometric_review.md"
        outfile.write_text(report)
        print(f"[SCAN] Report written to {outfile}")

        # Write JSON sidecar for deep-dive consumption
        if scored_papers:
            json_file = output_dir / f"{date_str}_scored.json"
            json_file.write_text(json.dumps(scored_papers, indent=2))
            print(f"[SCAN] JSON sidecar written to {json_file}")

        print(f"[SCAN] Papers with geometric relevance: {len(scored_papers)}")


if __name__ == "__main__":
    main()

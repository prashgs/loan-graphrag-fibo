from __future__ import annotations

"""Align GraphRAG entities/relationships to FIBO terms with Ollama.

This is a review-oriented mapping helper, not an OWL reasoner. It writes:
- output/fibo/fibo_alignment.json: detailed mappings and proposed enterprise extensions
- output/fibo/fibo_alignment_entities.csv: entity-level review table
- output/fibo/fibo_alignment_relationships.csv: relationship-level review table
"""

import csv
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "fibo"
TERMS_PATH = ROOT / "data" / "fibo_terms.json"
load_dotenv(ROOT / ".env")

OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b")
MIN_LEXICAL_SCORE = 68
MAX_TERMS_IN_PROMPT = 100

ENTITY_FILES = [
    "create_final_entities.parquet", "entities.parquet",
    "create_final_entities.csv", "entities.csv",
]
RELATIONSHIP_FILES = [
    "create_final_relationships.parquet", "relationships.parquet",
    "create_final_relationships.csv", "relationships.csv",
]


def read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read {path.name}. Install a parquet engine (e.g. `uv add pyarrow`) "
            f"and confirm the GraphRAG output schema. Details: {exc}"
        ) from exc


def find_artifact(names: list[str]) -> Path | None:
    for name in names:
        candidate = OUTPUT / name
        if candidate.exists():
            return candidate
    # Search nested output directories, preserving the preferred filename order.
    for name in names:
        matches = sorted(OUTPUT.rglob(name))
        if matches:
            return matches[0]
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def ollama_json(prompt: str) -> dict[str, Any] | None:
    """Call Ollama's local /api/chat endpoint using JSON mode."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content":
             "You are a cautious ontology-alignment assistant. Use only supplied candidates and evidence. "
             "Never invent an IRI. If no candidate is semantically appropriate, choose ENTERPRISE_EXTENSION. "
             "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result.get("message", {}).get("content", "")
        return json.loads(content)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def term_candidates(terms: list[dict[str, Any]], text: str, limit: int = 8) -> list[dict[str, Any]]:
    keys: list[str] = []
    key_to_term: dict[str, dict[str, Any]] = {}
    for term in terms:
        names = [term.get("label", ""), term.get("local_name", "")]
        names.extend(term.get("synonyms", []))
        for name in names:
            name = as_text(name)
            if name and name not in key_to_term:
                keys.append(name)
                key_to_term[name] = term
    found = process.extract(text, keys, scorer=fuzz.WRatio, limit=limit)
    result = []
    seen_iris = set()
    for label, score, _ in found:
        term = key_to_term[label]
        if term["iri"] in seen_iris:
            continue
        seen_iris.add(term["iri"])
        result.append({
            "label": term.get("label", ""),
            "iri": term["iri"],
            "definition": term.get("definition", ""),
            "kinds": term.get("kinds", []),
            "lexical_score": score,
        })
    return result


def choose_mapping(
    *, name: str, description: str, terms: list[dict[str, Any]], kind: str,
    source_text: str = "",
) -> dict[str, Any]:
    candidates = term_candidates(terms, name, limit=MAX_TERMS_IN_PROMPT)
    # Include description terms in ranking when entity name is an implementation-specific phrase.
    desc_candidates = term_candidates(terms, description, limit=12) if description else []
    by_iri: dict[str, dict[str, Any]] = {c["iri"]: c for c in candidates}
    for candidate in desc_candidates:
        if candidate["iri"] not in by_iri:
            by_iri[candidate["iri"]] = candidate
    candidates = list(by_iri.values())
    candidates.sort(key=lambda c: max(
        fuzz.WRatio(norm(name), norm(c["label"])),
        fuzz.WRatio(norm(name), norm(c["iri"].rsplit("/", 1)[-1])),
        c.get("lexical_score", 0),
    ), reverse=True)
    candidates = candidates[:MAX_TERMS_IN_PROMPT]

    prompt = f"""Align one extracted loan-domain {kind} to FIBO.
Return exactly this JSON shape:
{{
  "mapping_type": "FIBO_EXACT | FIBO_CLOSE_MATCH | ENTERPRISE_EXTENSION | UNRESOLVED",
  "canonical_label": "label or proposed enterprise term",
  "fibo_iri": "exact IRI from candidates, or empty string",
  "candidate_iri": "best candidate IRI if not accepted, else empty string",
  "confidence": 0.0,
  "rationale": "brief explanation",
  "proposed_enterprise_type": "class/property name if extension, otherwise empty string",
  "suggested_relationship_type": "UPPER_SNAKE_CASE if this is a relationship, otherwise empty string"
}}

Extracted {kind} name: {name}
Description: {description}
Source evidence: {source_text[:1200]}

FIBO candidates (the only IRIs you may select):
{json.dumps(candidates, ensure_ascii=False)}

Rules:
- Use FIBO_EXACT only if the concept is semantically the same as a candidate.
- Use FIBO_CLOSE_MATCH if related but not identical; leave fibo_iri empty and set candidate_iri.
- Use ENTERPRISE_EXTENSION for operational/internal concepts not modeled by these candidates, e.g. API service, workflow step, institution-specific rule.
- Use UNRESOLVED if the evidence is insufficient.
- Never infer a numeric policy threshold or regulatory requirement.
- For a relationship, map to an existing FIBO property only if the semantics match; otherwise propose an enterprise relationship type.
- confidence is 0 to 1.
"""
    result = ollama_json(prompt)
    if result:
        valid_iris = {term["iri"] for term in terms}
        iri = as_text(result.get("fibo_iri"))
        candidate_iri = as_text(result.get("candidate_iri"))
        if iri and iri not in valid_iris:
            result["fibo_iri"] = ""
            result["mapping_type"] = "UNRESOLVED"
            result["rationale"] = "Model returned an IRI outside the supplied FIBO candidates; rejected."
        if candidate_iri and candidate_iri not in valid_iris:
            result["candidate_iri"] = ""
        result.setdefault("confidence", 0.0)
        result.setdefault("rationale", "")
        result.setdefault("canonical_label", name)
        result.setdefault("mapping_type", "UNRESOLVED")
        result.setdefault("proposed_enterprise_type", "")
        result.setdefault("suggested_relationship_type", "")
        result["method"] = "ollama_review_required"
        return result

    # Safe fallback: suggest the closest lexical match but do not automatically accept it.
    best = candidates[0] if candidates else None
    score = float(best.get("lexical_score", 0)) if best else 0.0
    if best and score >= MIN_LEXICAL_SCORE:
        return {
            "mapping_type": "FIBO_CLOSE_MATCH",
            "canonical_label": best["label"],
            "fibo_iri": "",
            "candidate_iri": best["iri"],
            "confidence": min(score / 100, 0.85),
            "rationale": "Ollama unavailable; lexical match only. Human semantic review required.",
            "proposed_enterprise_type": "",
            "suggested_relationship_type": "",
            "method": "lexical_fallback_review_required",
        }
    extension = re.sub(r"[^A-Za-z0-9]+", "", name.title()) or "UnresolvedConcept"
    rel_type = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return {
        "mapping_type": "ENTERPRISE_EXTENSION" if name else "UNRESOLVED",
        "canonical_label": name or "Unresolved",
        "fibo_iri": "",
        "candidate_iri": best["iri"] if best else "",
        "confidence": 0.0,
        "rationale": "Ollama unavailable and no sufficiently strong lexical match; review enterprise extension proposal.",
        "proposed_enterprise_type": extension if kind == "entity" else "",
        "suggested_relationship_type": rel_type if kind == "relationship" else "",
        "method": "lexical_fallback_review_required",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def main() -> None:
    if not TERMS_PATH.exists():
        raise SystemExit("FIBO terms missing. Run `uv run python scripts/prepare_ontology.py` first.")

    entity_path = find_artifact(ENTITY_FILES)
    relationship_path = find_artifact(RELATIONSHIP_FILES)
    if entity_path is None:
        raise SystemExit(
            "GraphRAG entity artifact not found under output/. Run `uv run graphrag index --root .` first. "
            "If your GraphRAG version uses another artifact name, add it to ENTITY_FILES."
        )

    terms = json.loads(TERMS_PATH.read_text(encoding="utf-8"))
    entities = read_table(entity_path)
    relationships = read_table(relationship_path) if relationship_path else []

    entity_rows = []
    for entity in entities:
        name = as_text(entity.get("title") or entity.get("name") or entity.get("entity"))
        if not name:
            continue
        description = as_text(entity.get("description"))
        mapping = choose_mapping(name=name, description=description, terms=terms, kind="entity", source_text=description)
        entity_rows.append({
            "source_entity": name,
            "source_description": description,
            "mapping_type": mapping.get("mapping_type"),
            "canonical_label": mapping.get("canonical_label"),
            "fibo_iri": mapping.get("fibo_iri"),
            "candidate_iri": mapping.get("candidate_iri"),
            "confidence": mapping.get("confidence"),
            "rationale": mapping.get("rationale"),
            "proposed_enterprise_type": mapping.get("proposed_enterprise_type"),
            "method": mapping.get("method"),
            "source_artifact": str(entity_path.relative_to(ROOT)),
        })

    relationship_rows = []
    for rel in relationships:
        source = as_text(rel.get("source") or rel.get("source_entity"))
        target = as_text(rel.get("target") or rel.get("target_entity"))
        description = as_text(rel.get("description"))
        rel_name = as_text(rel.get("title") or rel.get("relationship") or rel.get("type"))
        if not rel_name:
            rel_name = description or "RELATED_TO"
        mapping = choose_mapping(
            name=rel_name, description=description, terms=terms, kind="relationship",
            source_text=f"source={source}; target={target}; description={description}",
        )
        relationship_rows.append({
            "source_entity": source,
            "source_relationship_label": rel_name,
            "target_entity": target,
            "source_description": description,
            "mapping_type": mapping.get("mapping_type"),
            "canonical_label": mapping.get("canonical_label"),
            "fibo_iri": mapping.get("fibo_iri"),
            "candidate_iri": mapping.get("candidate_iri"),
            "confidence": mapping.get("confidence"),
            "rationale": mapping.get("rationale"),
            "proposed_enterprise_relationship": mapping.get("suggested_relationship_type"),
            "method": mapping.get("method"),
            "source_artifact": str(relationship_path.relative_to(ROOT)) if relationship_path else "",
        })

    extension_types = sorted({
        row["proposed_enterprise_type"] for row in entity_rows
        if row.get("mapping_type") == "ENTERPRISE_EXTENSION" and row.get("proposed_enterprise_type")
    })
    extension_relationships = sorted({
        row["proposed_enterprise_relationship"] for row in relationship_rows
        if row.get("mapping_type") == "ENTERPRISE_EXTENSION" and row.get("proposed_enterprise_relationship")
    })
    report = {
        "metadata": {
            "ontology": "FIBO LoanApplications",
            "ontology_source": "https://github.com/edmcouncil/fibo/blob/master/LOAN/LoansGeneral/LoanApplications.rdf",
            "llm_provider": "Ollama local API",
            "llm_model": OLLAMA_MODEL,
            "warning": "Mappings are proposals. Review semantic equivalence and source evidence before production use.",
            "entity_artifact": str(entity_path.relative_to(ROOT)),
            "relationship_artifact": str(relationship_path.relative_to(ROOT)) if relationship_path else None,
        },
        "entities": entity_rows,
        "relationships": relationship_rows,
        "proposed_enterprise_extensions": {
            "classes": extension_types,
            "relationship_types": extension_relationships,
        },
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "fibo_alignment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(OUTPUT / "fibo_alignment_entities.csv", entity_rows)
    write_csv(OUTPUT / "fibo_alignment_relationships.csv", relationship_rows)
    print(f"Aligned {len(entity_rows)} entities and {len(relationship_rows)} relationships.")
    print("Outputs:")
    print("  output/fibo/fibo_alignment.json")
    print("  output/fibo/fibo_alignment_entities.csv")
    print("  output/fibo/fibo_alignment_relationships.csv")
    print(f"Proposed enterprise classes: {', '.join(extension_types) or '(none)'}")
    print(f"Proposed enterprise relationships: {', '.join(extension_relationships) or '(none)'}")
    print("Review every mapping; no mapping is considered approved automatically.")


if __name__ == "__main__":
    main()

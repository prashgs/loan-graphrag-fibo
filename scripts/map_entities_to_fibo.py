from __future__ import annotations

import csv
import json
from pathlib import Path

from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
TERMS_PATH = ROOT / "data" / "fibo_terms.json"
OUTPUT_DIR = ROOT / "output"
OUTPUT_CSV = OUTPUT_DIR / "fibo_mapping_candidates.csv"

# GraphRAG output artifact names can change across versions. Search common entity-table locations.
ENTITY_CANDIDATES = [
    ROOT / "output" / "entities.parquet",
    ROOT / "output" / "entities.csv",
    ROOT / "output" / "create_final_entities.parquet",
    ROOT / "output" / "create_final_entities.csv",
]


def read_entities(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}. Install pyarrow if needed, and check GraphRAG output schema: {exc}"
        ) from exc


def main() -> None:
    if not TERMS_PATH.exists():
        raise SystemExit("FIBO terms not found. Run: uv run python scripts/prepare_ontology.py")
    entity_path = next((path for path in ENTITY_CANDIDATES if path.exists()), None)
    if entity_path is None:
        raise SystemExit(
            "No GraphRAG entities file found in output/. Run `uv run graphrag index --root .` first. "
            "Inspect output/ and add its entity artifact path to ENTITY_CANDIDATES if your GraphRAG version differs."
        )

    terms = json.loads(TERMS_PATH.read_text(encoding="utf-8"))
    entities = read_entities(entity_path)

    searchable = []
    term_by_search = {}
    for term in terms:
        for candidate in {term.get("label", ""), term.get("local_name", ""), *term.get("synonyms", [])}:
            candidate = candidate.strip()
            if candidate:
                searchable.append(candidate)
                term_by_search[candidate] = term

    OUTPUT_DIR.mkdir(exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "graphrag_entity", "graphrag_description", "candidate_fibo_label",
            "candidate_fibo_iri", "candidate_definition", "similarity_score",
            "mapping_status", "source_artifact",
        ])
        writer.writeheader()
        for entity in entities:
            name = str(entity.get("title") or entity.get("name") or entity.get("entity") or "").strip()
            if not name:
                continue
            match = process.extractOne(name, searchable, scorer=fuzz.WRatio)
            matched_term = term_by_search[match[0]] if match else {}
            score = float(match[1]) if match else 0.0
            writer.writerow({
                "graphrag_entity": name,
                "graphrag_description": entity.get("description", ""),
                "candidate_fibo_label": matched_term.get("label", ""),
                "candidate_fibo_iri": matched_term.get("iri", ""),
                "candidate_definition": matched_term.get("definition", ""),
                "similarity_score": score,
                "mapping_status": "REVIEW_REQUIRED" if score >= 70 else "NO_CLEAR_MATCH",
                "source_artifact": str(entity_path.relative_to(ROOT)),
            })
    print(f"Wrote candidate mappings to {OUTPUT_CSV}")
    print("All matches require human review; string similarity is not semantic validation.")


if __name__ == "__main__":
    main()

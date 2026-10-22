from __future__ import annotations

"""Compare GraphRAG entity and relationship outputs from FIBO and no-FIBO runs."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

ENTITY_NAMES = ("create_final_entities.parquet", "entities.parquet", "create_final_entities.csv", "entities.csv")
REL_NAMES = ("create_final_relationships.parquet", "relationships.parquet", "create_final_relationships.csv", "relationships.csv")


def load_rows(root: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    path = None
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            path = matches[0]
            break
    if path is None:
        raise FileNotFoundError(f"No recognized artifact ({', '.join(names)}) under {root}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    return pd.read_parquet(path).to_dict(orient="records")


def get(row: dict[str, Any], *keys: str) -> str:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and not pd.isna(value):
            return str(value).strip()
    return ""


def entity_key(row: dict[str, Any]) -> str:
    return get(row, "title", "name", "entity").casefold()


def relation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    source = get(row, "source", "source_name").casefold()
    target = get(row, "target", "target_name").casefold()
    label = get(row, "type", "relationship_type", "description", "label").casefold()
    return source, target, label


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fibo-dir", type=Path, default=Path("output/fibo"))
    parser.add_argument("--no-fibo-dir", type=Path, default=Path("output/no_fibo"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/comparison"))
    args = parser.parse_args()

    fibo_entities = load_rows(args.fibo_dir, ENTITY_NAMES)
    plain_entities = load_rows(args.no_fibo_dir, ENTITY_NAMES)
    fibo_rels = load_rows(args.fibo_dir, REL_NAMES)
    plain_rels = load_rows(args.no_fibo_dir, REL_NAMES)

    fe = {entity_key(r): r for r in fibo_entities if entity_key(r)}
    pe = {entity_key(r): r for r in plain_entities if entity_key(r)}
    entity_rows: list[dict[str, Any]] = []
    for key in sorted(fe.keys() | pe.keys()):
        if key in fe and key in pe:
            status = "common"
        elif key in fe:
            status = "only_fibo_aware"
        else:
            status = "only_ontology_agnostic"
        row = {"entity": get(fe.get(key, {}), "title", "name") or get(pe.get(key, {}), "title", "name"), "status": status,
               "fibo_type": get(fe.get(key, {}), "type"), "no_fibo_type": get(pe.get(key, {}), "type"),
               "fibo_description": get(fe.get(key, {}), "description"), "no_fibo_description": get(pe.get(key, {}), "description")}
        entity_rows.append(row)

    fr = {relation_key(r): r for r in fibo_rels if relation_key(r) != ("", "", "")}
    pr = {relation_key(r): r for r in plain_rels if relation_key(r) != ("", "", "")}
    relation_rows: list[dict[str, Any]] = []
    for key in sorted(fr.keys() | pr.keys()):
        status = "common" if key in fr and key in pr else ("only_fibo_aware" if key in fr else "only_ontology_agnostic")
        row = {"source": key[0], "target": key[1], "relationship_label": key[2], "status": status,
               "fibo_description": get(fr.get(key, {}), "description"), "no_fibo_description": get(pr.get(key, {}), "description")}
        relation_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_write(args.output_dir / "entity_differences.csv", entity_rows)
    csv_write(args.output_dir / "relationship_differences.csv", relation_rows)
    summary = {
        "fibo_aware_entity_count": len(fe), "ontology_agnostic_entity_count": len(pe),
        "common_entity_count": len(fe.keys() & pe.keys()),
        "only_fibo_aware_entity_count": len(fe.keys() - pe.keys()),
        "only_ontology_agnostic_entity_count": len(pe.keys() - fe.keys()),
        "fibo_aware_relationship_count": len(fr), "ontology_agnostic_relationship_count": len(pr),
        "common_relationship_count": len(fr.keys() & pr.keys()),
        "only_fibo_aware_relationship_count": len(fr.keys() - pr.keys()),
        "only_ontology_agnostic_relationship_count": len(pr.keys() - fr.keys()),
        "note": "Exact normalized labels/endpoints are compared; semantically equivalent renamed entities may appear as differences. Review before drawing conclusions."
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

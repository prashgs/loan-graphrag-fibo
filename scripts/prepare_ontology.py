from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_DIR = ROOT / "ontology"
DATA_DIR = ROOT / "data"
RDF_URL = "https://raw.githubusercontent.com/edmcouncil/fibo/master/LOAN/LoansGeneral/LoanApplications.rdf"
RDF_PATH = ONTOLOGY_DIR / "LoanApplications.rdf"
OUTPUT_PATH = DATA_DIR / "fibo_terms.json"


def local_name(iri: str) -> str:
    return iri.rstrip("/#").rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def first_literal(graph: Graph, subject: URIRef, predicates: list[URIRef]) -> str:
    for predicate in predicates:
        values = list(graph.objects(subject, predicate))
        if values:
            return str(values[0])
    return ""


def main() -> None:
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RDF_PATH.exists():
        req = Request(RDF_URL, headers={"User-Agent": "loan-graphrag-fibo-starter/0.1"})
        with urlopen(req, timeout=30) as response:
            RDF_PATH.write_bytes(response.read())
        print(f"Downloaded {RDF_PATH}")

    graph = Graph()
    graph.parse(RDF_PATH, format="xml")

    term_types = [
        (OWL.Class, "class"),
        (RDFS.Class, "class"),
        (OWL.ObjectProperty, "object_property"),
        (OWL.DatatypeProperty, "datatype_property"),
        (OWL.NamedIndividual, "individual"),
    ]
    terms: dict[str, dict] = {}

    for rdf_type, kind in term_types:
        for subject in graph.subjects(RDF.type, rdf_type):
            if not isinstance(subject, URIRef):
                continue
            iri = str(subject)
            label = first_literal(graph, subject, [RDFS.label, SKOS.prefLabel])
            if not label:
                label = local_name(iri)
            definition = first_literal(
                graph, subject, [SKOS.definition, RDFS.comment, URIRef("http://purl.org/dc/terms/abstract")]
            )
            synonyms = sorted({
                str(value) for predicate in (SKOS.altLabel, URIRef("https://www.omg.org/spec/Commons/AnnotationVocabulary/synonym"))
                for value in graph.objects(subject, predicate)
            })
            entry = terms.setdefault(iri, {
                "iri": iri, "label": label, "local_name": local_name(iri),
                "kinds": [], "definition": definition, "synonyms": synonyms,
            })
            if kind not in entry["kinds"]:
                entry["kinds"].append(kind)
            if not entry["definition"] and definition:
                entry["definition"] = definition

    payload = sorted(terms.values(), key=lambda item: (item["label"].lower(), item["iri"]))
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Parsed {len(payload)} FIBO terms -> {OUTPUT_PATH}")
    print("Note: this script parses the LoanApplications module only; it does not resolve every imported ontology.")
    print("For production, pin a FIBO commit/release and load import dependencies if full OWL reasoning is required.")


if __name__ == "__main__":
    main()

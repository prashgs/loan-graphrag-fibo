# Loan GraphRAG: FIBO-aware vs. ontology-agnostic extraction with Ollama

A local-first starter repository using **Microsoft GraphRAG** to extract a graph from sample loan-application text and **Ollama** for local LLM/embedding calls. It supports two independent runs over the same corpus:

- **FIBO-aware extraction** uses FIBO LoanApplications vocabulary as guidance.
- **Ontology-agnostic extraction** uses only the source text and does not consult FIBO vocabulary.

The sample corpus is a placeholder for a future Confluence export. This repository does **not** connect to Confluence.

## What it does

1. Downloads the official FIBO `LoanApplications.rdf` module and parses terms into `data/fibo_terms.json`.
2. Runs GraphRAG separately with the FIBO-aware and ontology-agnostic extraction prompts, keeping output, cache, and logs in separate directories.
3. Compares extracted entity and relationship labels/endpoints and reports common items and items unique to each run.
4. Optionally maps the FIBO-aware graph to FIBO IRIs and proposes enterprise extensions for concepts that do not fit the ontology.

> GraphRAG's extracted graph is not itself an OWL reasoner. FIBO mappings and differences are candidate analysis results and should be reviewed by a domain expert.

## Requirements

- Python 3.10–3.12 (check the current GraphRAG release requirements)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- Enough RAM/VRAM for your selected model

## Quick start

```bash
uv sync
cp .env.example .env
ollama serve
ollama pull qwen3.5:4b
ollama pull nomic-embed-text

# Optional: download/parse the FIBO module for the later alignment step
uv run python scripts/prepare_ontology.py

# Run both extraction variants against the same input/*.txt corpus.
# Each variant has its own output, cache, and logs.
uv run graphrag index --root . --config settings.fibo.yaml
uv run graphrag index --root . --config settings.no_fibo.yaml

# Compare the resulting graphs
uv run python scripts/compare_extractions.py

# Optional: align the FIBO-aware extraction to FIBO IRIs / proposed enterprise extensions
uv run python scripts/align_graph_to_fibo.py
```

If your installed GraphRAG CLI uses different option names, check `uv run graphrag index --help`. GraphRAG config and output schemas can vary between releases; verify the generated artifacts before interpreting comparison results.

## Interpreting the comparison

`output/comparison/summary.json` reports counts and overlap. The CSV files provide detail:

- `output/comparison/entity_differences.csv`
- `output/comparison/relationship_differences.csv`

Statuses are `common`, `only_fibo_aware`, and `only_ontology_agnostic`. Matching is based on normalized names and relationship endpoints/labels. If FIBO guidance causes a model to rename an entity or relationship, the comparison may treat those as different even when they are semantically similar. For deeper evaluation, add a human-reviewed synonym/semantic-equivalence pass. Counts alone do not prove one mode is better.

## Configurations

- `settings.fibo.yaml`: ontology-guided extraction; output/cache/logs under `output/fibo`, `cache/fibo`, and `logs/fibo`.
- `settings.no_fibo.yaml`: ontology-agnostic extraction; output/cache/logs under `output/no_fibo`, `cache/no_fibo`, and `logs/no_fibo`.
- `settings.yaml`: backward-compatible default pointing to the FIBO-aware configuration.

Both runs use the same model, input corpus, chunking, and entity-type list where possible. The ontology-agnostic prompt instructs the model not to use FIBO and allows source terminology to drive entity labels. Note that any fixed `entity_types` list in the config can itself constrain extraction; for a stricter unconstrained experiment, remove `extract_graph.entity_types` from `settings.no_fibo.yaml` if supported by your installed GraphRAG version.

## FIBO alignment outputs

After the FIBO-aware indexing run and ontology preparation, `scripts/align_graph_to_fibo.py` attempts to read GraphRAG artifacts and writes review files under `output/`. The script calls local Ollama and falls back to lexical suggestions if Ollama is unavailable. It only accepts IRIs present in the parsed term list; the term list is based on `LoanApplications.rdf` and does not automatically load all imported OWL dependencies.

## Using a different corpus

Put `.txt` files in `input/` or adjust the input file pattern in both config files. Use synthetic/non-sensitive sample text for development. Avoid using actual customer loan records unless privacy, access-control, retention, and compliance requirements have been reviewed.

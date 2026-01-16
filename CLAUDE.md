# WakaDecoder Project

## Overview
Classical Japanese poetry learning platform for Chinese speakers. Build pipeline (offline, LLM-assisted) generates lessons from corpora; runtime serves pre-generated content.

## Key Documentation
- `WakaDecoder_PROMPT_v2.md` — Full specification
- `WakaDecoder_PROMPT_v2_PROMPTS.md` — LLM prompts & Pydantic schemas
- `WakaDecoder_RESOURCES.md` — Corpus acquisition rules
- `SCAFFOLD_PLAN.md` — Phase-by-phase build plan

## Quick Start
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
pip install -r requirements.txt
```

## Constraints (Non-Negotiable)
- **No hardcoded API keys** — all credentials in `.env`
- **No example/placeholder text in UI** — only real data from classroom.db
- **All scraping in offline scripts only** — never in Streamlit runtime
- **Modular architecture** — UI in `app.py`, logic in `wakadecoder/`
- **Fugashi is token source of truth** — LLM doesn't invent tokens
- **Two-level grammar IDs** — `canonical_id` + `sense_id`

## Architecture
```
wakadecoder/
├── app.py                 # Streamlit entry point (UI only)
├── wakadecoder/           # Core logic
│   ├── schemas/           # Pydantic models
│   ├── classroom/         # Runtime: loader, progress, navigator
│   ├── viewer/            # Runtime: lesson, quiz, reference rendering
│   └── optional/          # Live tutor (disabled by default)
├── scripts/               # Build pipeline (offline)
│   ├── 01_ingest_corpus.py
│   ├── 02_annotate_corpus.py
│   ├── 03_extract_curriculum.py
│   ├── 04_generate_lessons.py
│   └── 05_compile_classroom.py
├── prompts/               # LLM prompt templates (YAML)
├── data/                  # Generated data (gitignored)
└── tests/                 # Validation tests
```

## Build Pipeline Order
```
01_ingest → 02_annotate → 03_extract_curriculum → 04_generate_lessons → 05_compile_classroom
```
Each script has `--help`. Run sequentially; each depends on previous output.

## Current Phase
Check `SCAFFOLD_PLAN.md` for current implementation status.

## Testing Checkpoints
After each phase, verify:
- Phase 1: `python -c "from wakadecoder.schemas import *"` works
- Phase 2: `python scripts/01_ingest_corpus.py --help` works
- Phase 3: `data/annotated/poems.parquet` exists and validates
- Phase 4: `data/curriculum/lesson_graph.json` has no cycles
- Phase 5: `data/lessons/*.json` all validate against schema
- Phase 6: `streamlit run app.py` loads without errors

## LLM Usage
- **Build time**: Gemini 3 Flash for annotation + lesson generation
- **Runtime**: No LLM by default; optional live tutor module

## Git Workflow
- Commit after completing each phase
- Never commit `.env`, `data/`, or `__pycache__/`
- Use conventional commits: `feat:`, `fix:`, `docs:`

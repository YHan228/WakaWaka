# WakaWaka Scaffold Plan

## Build Phases Overview

| Phase | Name | Output | Est. Complexity |
|-------|------|--------|-----------------|
| 0 | Project Setup | requirements.txt, .gitignore, directory structure | Low |
| 1 | Schemas | `wakawaka/schemas/*.py` | Medium |
| 2 | Prompts | `prompts/*.yaml` | Low |
| 3 | Ingest Script | `scripts/01_ingest_corpus.py` | Medium |
| 4 | Annotate Script | `scripts/02_annotate_corpus.py` | High |
| 5 | Curriculum Script | `scripts/03_extract_curriculum.py` | High |
| **5.5** | **Corpus Analysis** | **Review & validate before lesson generation** | **CHECKPOINT** |
| 6 | Lesson Gen Script | `scripts/04_generate_lessons.py` | High |
| 7 | Compile Script | `scripts/05_compile_classroom.py` | Medium |
| 8 | Runtime Core | `wakawaka/classroom/*.py` | Medium |
| 9 | Runtime Viewer | `wakawaka/viewer/*.py` | Medium |
| 10 | Streamlit App | `app.py` | Medium |
| 11 | Integration Test | End-to-end with sample data | Medium |

---

## Phase 0: Project Setup
**Goal**: Initialize project structure and dependencies.

### Tasks
- [ ] Create directory structure
- [ ] Create `requirements.txt`
- [ ] Create `.gitignore`
- [ ] Initialize git repo (if not done)
- [ ] Verify `.env` is set up

### Files to Create
```
wakawaka/
├── wakawaka/
│   ├── __init__.py
│   ├── schemas/
│   │   └── __init__.py
│   ├── classroom/
│   │   └── __init__.py
│   ├── viewer/
│   │   └── __init__.py
│   └── optional/
│       └── __init__.py
├── scripts/
│   └── __init__.py
├── prompts/
├── data/
│   ├── raw/
│   ├── annotated/
│   ├── curriculum/
│   └── lessons/
├── tests/
│   └── __init__.py
├── requirements.txt
└── .gitignore
```

### requirements.txt
```
# Build pipeline
fugashi>=1.3.0
unidic-lite>=1.0.8
pandas>=2.0.0
polars>=0.20.0
google-generativeai>=0.3.0
networkx>=3.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0
pyarrow>=14.0.0

# Runtime
streamlit>=1.30.0

# Testing
pytest>=7.0.0
```

### .gitignore
```
.env
data/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
.streamlit/
```

### Verification
```bash
pip install -r requirements.txt
python -c "import fugashi, pandas, google.generativeai, networkx, pydantic, streamlit"
```

---

## Phase 1: Schemas
**Goal**: Define all Pydantic models from `WakaWaka_PROMPT_v2_PROMPTS.md`.

### Tasks
- [ ] Create `wakawaka/schemas/annotation.py`
- [ ] Create `wakawaka/schemas/curriculum.py`
- [ ] Create `wakawaka/schemas/lesson.py`
- [ ] Create `wakawaka/schemas/progress.py`
- [ ] Create `wakawaka/schemas/__init__.py` (exports all)
- [ ] Write schema validation tests

### Verification
```bash
python -c "from wakawaka.schemas import PoemAnnotation, LessonGraph, LessonContent, StudentProgress"
pytest tests/test_schemas.py
```

---

## Phase 2: Prompts
**Goal**: Create YAML prompt templates.

### Tasks
- [ ] Create `prompts/annotate.yaml`
- [ ] Create `prompts/generate_lesson.yaml`
- [ ] Create `prompts/live_tutor.yaml`
- [ ] Create `wakawaka/utils/prompt_loader.py`

### Verification
```bash
python -c "from wakawaka.utils.prompt_loader import load_prompt; p = load_prompt('annotate'); print(p['meta']['version'])"
```

---

## Phase 3: Ingest Script
**Goal**: Fetch raw corpus data with native Japanese script.

### Critical Note: Script Type Requirements
| Source | Script Type | Use Case |
|--------|-------------|----------|
| ogura100 | Native Japanese (hiragana/kanji) | Display, tokenization, lessons |
| lapis | Native Japanese | Display, tokenization, lessons |
| oncoj | **Romanized Old Japanese** | Grammar analysis only (NOT for display) |

**ONCOJ limitation**: Extracts PHON nodes which are romanized transcriptions (e.g., "miatotukuru..."). Native Man'yōgana exists in the corpus but requires complex extraction. For MVP, use Lapis for native script poems.

### Tasks
- [x] Implement ONCOJ tree parser (Penn Treebank format) — outputs romanized
- [x] Create `scripts/01_ingest_corpus.py` with CLI
- [x] Implement ogura100 source (100 poems, native script)
- [x] **Implement Lapis scraper** (1000 poems, native script) — **COMPLETE**
- [x] Handle rate limiting, caching, provenance

### Target Corpus Size
- **Minimum**: 300 poems in native Japanese script
- **Recommended**: 500-1000 poems for good grammar coverage
- **Current: 1100 (100 ogura100 + 1000 lapis) — excellent ✓**

### CLI Interface
```bash
# Native script sources (for lessons)
python scripts/01_ingest_corpus.py --source ogura100
python scripts/01_ingest_corpus.py --source lapis --max-poems 500

# Romanized (for grammar analysis reference only)
python scripts/01_ingest_corpus.py --source oncoj --max-poems 500

# All sources
python scripts/01_ingest_corpus.py --source all --max-poems 1000
```

### Verification
```bash
# Check native script count (should be 300+)
grep -c '"source": "ogura100"\|"source": "lapis"' data/raw/poems.jsonl

# Verify text is native Japanese (not romanized)
head -3 data/raw/poems.jsonl | grep -o '"text": "[^"]*"' | head -1
```

---

## Phase 4: Annotate Script
**Goal**: LLM-annotate poems with grammar points, vocabulary, difficulty.

### Key Design Decisions (from main prompt)
1. **Fugashi is source of truth for tokens** — LLM does not invent tokens, only provides readings
2. **Two-level grammar IDs** — `canonical_id` (e.g., "particle_ni") + `sense_id` (e.g., "location")
3. **Deterministic difficulty scoring** — computed from factors using formula: `1 - Π(1-weight_i)`
4. **Span convention** — all spans are `[start, end)`, 0-based, end-exclusive (Python slice semantics)
5. **Text hash** — SHA256 prefix stored for integrity verification
6. **Token readings** — aligned to Fugashi tokens (enables deterministic ruby generation)
7. **Vocabulary extraction** — words Chinese speakers need to learn, with cognate notes

### Tasks
- [ ] Implement Fugashi tokenization with spans (token source of truth)
- [ ] Implement text hash generation (SHA256 prefix)
- [ ] Create Gemini API client wrapper
- [ ] Implement LLM prompt for: readings, grammar_points (two-level), vocabulary, difficulty_factors
- [ ] Implement post-processing: compute difficulty_score_computed from factors
- [ ] Validate spans against text length
- [ ] Verify token_readings count matches Fugashi token count
- [ ] Implement batch processing with checkpointing (resume on failure)
- [ ] Cache LLM responses keyed by (poem_id, model, prompt_version)
- [ ] Create `scripts/02_annotate_corpus.py`
- [ ] Validate against PoemAnnotation schema (retry malformed responses once)

### CLI Interface
```bash
python scripts/02_annotate_corpus.py --input-dir data/raw --output data/annotated/poems.parquet --batch-size 10 --model gemini-2.0-flash --resume
```

### Verification
```bash
python scripts/02_annotate_corpus.py --input-dir data/raw --output data/annotated/poems.parquet --batch-size 5
python -c "
import pandas as pd
from wakawaka.schemas import PoemAnnotation
df = pd.read_parquet('data/annotated/poems.parquet')
print(f'Poems: {len(df)}')
# Validate first row
row = df.iloc[0].to_dict()
PoemAnnotation(**row)
print('Schema valid!')
"
```

---

## Phase 5: Curriculum Extraction Script
**Goal**: Derive lesson graph from annotation statistics — no hardcoding.

### Key Design Decisions (from main prompt)
1. **Two-level identity** — lessons driven by `canonical_id`; senses introduced progressively
2. **Prerequisite stoplist** — ultra-common particles (は, の, を, に, が) excluded from prerequisites
3. **Cycle detection** — uses networkx; breaks cycles by removing weakest edge
4. **Multi-criteria prerequisites** — requires: `co_ratio ≥70%`, `difficulty_gap ≥0.05`, `support ≥3 poems`

### Tasks
- [ ] Implement two-level grammar index building (group by canonical_id, aggregate senses)
- [ ] Implement co-occurrence matrix computation (canonical level only — avoids ID explosion)
- [ ] Implement prerequisite inference with multi-criteria thresholds
- [ ] Implement stoplist filtering (STOPLIST_CANONICAL_IDS)
- [ ] Implement cycle detection using networkx
- [ ] Implement cycle breaking (remove weakest edge by difficulty_gap, then co_ratio)
- [ ] Implement topological sort and grouping into units
- [ ] Assign poems to lessons (sorted by difficulty)
- [ ] Create `scripts/03_extract_curriculum.py`
- [ ] Generate curriculum_report.md summarizing discovered structure

### CLI Interface
```bash
python scripts/03_extract_curriculum.py --input data/annotated/poems.parquet --output-dir data/curriculum --min-poems-per-lesson 3 --max-lessons 50 --difficulty-tiers 5
```

### Output Files
- `data/curriculum/grammar_index.json` — discovered grammar points (canonical + senses)
- `data/curriculum/lesson_graph.json` — lessons with prerequisites + cycle detection metadata
- `data/curriculum/prerequisite_graph.json` — raw prerequisite edges for analysis
- `data/curriculum/curriculum_report.md` — human-readable summary

### Verification
```bash
python scripts/03_extract_curriculum.py --input data/annotated/poems.parquet --output-dir data/curriculum
python -c "
import json
from wakawaka.schemas import LessonGraph
with open('data/curriculum/lesson_graph.json') as f:
    data = json.load(f)
graph = LessonGraph(**data)
print(f'Units: {len(graph.units)}')
print(f'Lessons: {graph.meta[\"total_lessons\"]}')
print(f'Cycles broken: {graph.meta[\"cycles_broken\"]}')
"
```

---

## Phase 5.5: Corpus Analysis & Validation (CHECKPOINT)
**Goal**: Review annotated corpus and curriculum before spending LLM tokens on lesson generation.

**STOP HERE AND REVIEW** — Do not proceed to Phase 6 until you're satisfied with the data.

### Tasks
- [ ] Create `scripts/analyze_corpus.py` — generates comprehensive report
- [ ] Review annotation quality (sample 10-20 poems manually)
- [ ] Verify grammar point distribution makes sense
- [ ] Check difficulty score distribution
- [ ] Review lesson graph for sensible prerequisites
- [ ] Approve or re-run annotation with adjusted prompts

### Analysis Script Output
```bash
python scripts/analyze_corpus.py --poems data/annotated/poems.parquet --curriculum data/curriculum --output data/analysis/
```

### Generated Reports
```
data/analysis/
├── corpus_summary.md           # Overview stats
├── grammar_point_frequency.csv # canonical_id, sense_id, count, examples
├── difficulty_distribution.png # Histogram
├── vocabulary_coverage.csv     # Words extracted, by frequency
├── lesson_graph.png            # Visual DAG of prerequisites
├── sample_annotations.md       # 10 random poems with full annotations
└── quality_flags.md            # Potential issues detected
```

### What to Check

**1. Corpus Summary**
- Total poems, sources, date range
- Average tokens per poem
- Annotation coverage (% with grammar points, vocabulary)

**2. Grammar Point Frequency**
```
| canonical_id    | sense_id | count | avg_difficulty | example_surface |
|-----------------|----------|-------|----------------|-----------------|
| particle_wa     | topic    | 342   | 0.15           | は              |
| particle_ni     | location | 156   | 0.22           | に              |
| auxiliary_keri  | past     | 89    | 0.45           | けり            |
```
- Are frequencies reasonable?
- Any garbage IDs (typos, malformed)?
- Missing expected grammar points?

**3. Difficulty Distribution**
- Should be roughly normal or slightly right-skewed
- Flag if >50% poems have difficulty <0.1 or >0.9
- Check poems at extremes — do scores make sense?

**4. Vocabulary Extraction**
- Are vocabulary items actually useful for Chinese speakers?
- Are obvious cognates correctly skipped?
- Sample meanings — are they accurate?

**5. Lesson Graph**
- Does the prerequisite order make sense pedagogically?
- Are there suspicious edges (e.g., rare point as prereq of common)?
- How many cycles were broken? Review removed_edges.

**6. Sample Annotations (Manual Review)**
- Read 10-20 poems with their full annotations
- Check: spans correct? readings accurate? grammar IDs sensible?
- Note any systematic errors to fix in prompt

### Quality Flags (Auto-detected)
The script should flag:
- [ ] Poems with 0 grammar points
- [ ] Poems with 0 vocabulary items
- [ ] Grammar IDs that appear only once (likely typos)
- [ ] Spans that extend beyond text length
- [ ] Token reading counts that don't match Fugashi token counts
- [ ] Difficulty scores of exactly 0.0 or 1.0 (suspicious)

### Decision Point
After review, choose one:

| Finding | Action |
|---------|--------|
| Looks good | Proceed to Phase 6 |
| Minor issues | Fix in post-processing, proceed |
| Systematic annotation errors | Adjust prompt, re-run Phase 4 |
| Corpus too small/biased | Go back to Phase 3, add more sources |

### Verification
```bash
python scripts/analyze_corpus.py --poems data/annotated/poems.parquet --curriculum data/curriculum --output data/analysis/
cat data/analysis/corpus_summary.md
cat data/analysis/quality_flags.md
# Open data/analysis/lesson_graph.png in image viewer
# Manually review data/analysis/sample_annotations.md
```

---

## Phase 5.5b: LLM Curriculum Refinement (Optional)
**Goal**: Use LLM to reorganize algorithmic curriculum into pedagogically-sound units.

The algorithmic curriculum (Phase 5) groups by grammar category. LLM refinement creates thematic units with meaningful prerequisites.

### Key Design Decisions
1. **Strict output schema** — LLM outputs only valid JSON with exact lesson IDs
2. **Validation** — All lesson IDs verified against original; prerequisites must reference earlier units
3. **Non-destructive** — Original curriculum preserved; refined output to separate directory
4. **Ensemble mode** — Run multiple independent trials, then synthesize best curriculum

### Prompts (in `prompts/`)
- `curriculum_trial.yaml` — Individual trial prompt (higher temperature for diversity)
- `curriculum_synthesis.yaml` — Synthesis prompt to combine trials

### Script
```bash
# Single-shot refinement
python scripts/03b_refine_curriculum_llm.py

# Ensemble mode with parallel execution (recommended)
python scripts/03b_refine_curriculum_llm.py --ensemble 10 --parallel 3
python scripts/03b_refine_curriculum_llm.py --ensemble 10 --parallel 5 --model gemini-2.5-pro
```

### Input/Output
- Input: `data/curriculum/` (algorithmic)
- Output: `data/curriculum_refined/` (LLM-designed)
  - `lesson_graph.json` — reorganized units with titles and prerequisites
  - `llm_refinements.json` — raw LLM response (or synthesis result)
  - `curriculum_report.md` — human-readable summary
  - `lesson_context.yaml` — contextual guidance for lesson generation (NEW)
  - `trials/` — (ensemble mode) individual trial proposals

### Lesson Context Generation
After synthesis, the judge LLM generates `lesson_context.yaml` containing:
- Curriculum philosophy and progression logic
- Per-unit teaching emphasis and context
- Grammar-specific teaching notes
- Cross-cutting pedagogical guidelines

This context is automatically loaded by `04_generate_lessons.py` to maintain consistency.

### LLM Prompt Constraints
- Provide exact list of valid lesson IDs
- Require unit IDs: `unit_01`, `unit_02`, ...
- Prerequisites must reference lessons in earlier units only
- JSON-only output, no explanatory text

### Verification
```bash
python scripts/03b_refine_curriculum_llm.py --ensemble 10 --parallel 3
cat data/curriculum_refined/curriculum_report.md
cat data/curriculum_refined/lesson_context.yaml  # Check generated context
ls data/curriculum_refined/trials/  # See individual trial proposals
```

---

## Phase 6: Lesson Generation Script
**Goal**: Generate complete lesson content using LLM.

### Two-Step Generation Process
1. **Poem Selection** (optional, via `--select-poems`): LLM selects 2-3 best teaching poems from candidate pool of 10
2. **Lesson Generation**: LLM generates full lesson content using selected poems

### Input Files
- `data/curriculum_refined/lesson_graph.json` — LLM-refined curriculum (preferred)
- `data/curriculum/lesson_graph.json` — algorithmic curriculum (fallback)
- `data/annotated/poems.parquet` — poems with full annotations

### Curriculum Structure (Phase 5 output)
Each lesson now contains:
- `candidate_poem_ids` — pool of 10 poems (algorithmically selected by difficulty)
- `poem_ids` — final selected poems (empty initially, populated by LLM selection step)

### Prompts (in `prompts/`)
- `select_poems.yaml` — LLM poem selection from candidates
- `generate_lesson.yaml` — Full lesson content generation

### LLM Generates (per lesson)
- `lesson_title` — human-readable title
- `lesson_summary` — 1-2 sentences
- `grammar_explanation` — concept, formation, variations, common_confusions
- `teaching_sequence` — ordered steps: introduction, poem_presentation (with **complete vocabulary for every word**, focus_highlight), grammar_spotlight, contrast_example, comprehension_check, summary
- `reference_card` — quick reference for review (point, one_liner, example, see_also)
- `forward_references` — grammar in poems not yet taught (e.g., "We'll learn けり in Unit 3")

### Tasks
- [x] Create lesson generation prompt templating (use prompts/generate_lesson.yaml)
- [x] Create poem selection prompt (use prompts/select_poems.yaml)
- [x] Provide context to LLM: grammar_point metadata, assigned poems, prerequisite summaries
- [x] Include student level assumption: "Chinese native speaker; knows hiragana, leverages kanji from Chinese, no Japanese grammar"
- [x] Implement batch lesson generation with checkpointing
- [x] Create `scripts/04_generate_lessons.py`
- [x] Validate against LessonContent schema
- [x] Generate `lessons_manifest.json` listing all generated lessons

### CLI Interface
```bash
# Full pipeline with parallel generation (recommended)
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --all --select-poems --parallel 3 --resume

# Sequential generation (safer, easier to debug)
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --all --select-poems --resume

# Without LLM poem selection (uses first 3 candidates)
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --all --resume

# Generate specific lessons
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --lesson-ids lesson_particle_no,lesson_particle_ni --select-poems
```

### Lesson Context Integration
If `lesson_context.yaml` exists in the curriculum directory (generated by Phase 5.5b), it is automatically loaded and injected into the lesson generation prompt. This provides:
- Curriculum-wide teaching philosophy
- Unit-specific pedagogical guidance
- Grammar-specific teaching notes

### Output
- `data/lessons/{lesson_id}.json` — full lesson content
- `data/lessons/lessons_manifest.json` — list of all generated lessons

### Verification
```bash
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --all --select-poems --parallel 3 --max-lessons 5
python -c "
import json, glob
from wakawaka.schemas import LessonContent
for f in glob.glob('data/lessons/*.json')[:3]:
    if 'manifest' in f: continue
    with open(f) as fp:
        LessonContent(**json.load(fp))
    print(f'{f}: valid')
"
```

---

## Phase 7: Compile Script
**Goal**: Bundle everything into deployable classroom.db.

### SQLite Schema (from main prompt)
- `units` — id, title, position
- `lessons` — id, unit_id, title, summary, position, content (JSON), prerequisites (JSON)
- `poems` — id, text, reading_hiragana, reading_romaji, annotations (JSON)
- `grammar_points` — id, category, summary, reference_card (JSON), lesson_id

**Note**: Progress table is stored separately in `~/.wakawaka/progress.db` (user data, not content).

### Tasks
- [ ] Create SQLite schema (tables above)
- [ ] Implement data loading from lessons/*.json, curriculum/*.json, poems.parquet
- [ ] Implement integrity checks: all referenced poem_ids exist, all prerequisites exist
- [ ] Create `scripts/05_compile_classroom.py`
- [ ] Generate `classroom_stats.json`: total lessons, poems, estimated study hours

### CLI Interface
```bash
python scripts/05_compile_classroom.py --lessons data/lessons --curriculum data/curriculum --poems data/annotated/poems.parquet --output data/classroom.db
```

### Verification
```bash
python scripts/05_compile_classroom.py --lessons data/lessons --curriculum data/curriculum --poems data/annotated/poems.parquet --output data/classroom.db
sqlite3 data/classroom.db "SELECT COUNT(*) FROM lessons;"
sqlite3 data/classroom.db "SELECT COUNT(*) FROM poems;"
```

---

## Phase 8: Runtime Core
**Goal**: Implement classroom data loading and navigation.

### Tasks
- [ ] Create `wakawaka/classroom/loader.py`
- [ ] Create `wakawaka/classroom/progress.py`
- [ ] Create `wakawaka/classroom/navigator.py`

### Verification
```bash
python -c "
from wakawaka.classroom import ClassroomLoader, Navigator
loader = ClassroomLoader('data/classroom.db')
nav = Navigator(loader)
print(f'Total lessons: {nav.total_lessons}')
print(f'First lesson: {nav.get_first_lesson().lesson_title}')
"
```

---

## Phase 9: Runtime Viewer
**Goal**: Implement lesson rendering components.

### Tasks
- [ ] Create `wakawaka/viewer/lesson.py` (render lesson content)
- [ ] Create `wakawaka/viewer/quiz.py` (render comprehension checks)
- [ ] Create `wakawaka/viewer/reference.py` (grammar reference cards)
- [ ] Implement deterministic ruby generation from tokens
- [ ] Implement interactive vocabulary hover system

### Interactive Vocabulary Hover (UI Design)
Each word in poem text should be interactive:
- **Color coding by word type**: particles (blue), verbs (green), nouns (default), auxiliaries (purple)
- **Hover/tap popup**: Shows reading (hiragana) + meaning + Chinese cognate note
- **Span-based highlighting**: Uses `vocabulary[].span` to map words to text positions
- **Fallback**: Words without spans shown in separate vocabulary list below poem

Implementation approach:
1. Parse poem text and vocabulary spans
2. Generate HTML with `<span class="vocab-word" data-idx="N">` wrappers
3. Attach hover/click handlers to show tooltip with vocab info
4. Style with CSS: color by word type, hover highlight

### Verification
```bash
python -c "
from wakawaka.viewer import render_poem_with_ruby, render_lesson
# Test with mock data
"
```

---

## Phase 10: Streamlit App
**Goal**: Build the main UI.

### UI Requirements (from main prompt section 4)
- **Sidebar**: curriculum tree with progress indicators (✓ completed, → current, ○ available)
- **Main area**: lesson content renderer (markdown/HTML)
- **Navigation**: previous/next, jump to any unlocked lesson
- **Progress**: persists in `~/.wakawaka/progress.db` (separate from content)
- **No LLM calls** during normal lesson navigation

### Reference Mode (from main prompt section 4.2)
- Grammar point index (searchable)
- Reference cards for all learned points
- Poem anthology (browse all poems, grouped by collection)

### Tasks
- [ ] Create `app.py` with sidebar curriculum tree (progress indicators)
- [ ] Implement lesson view with poem display (furigana, romaji, translation, vocabulary)
- [ ] Implement navigation (previous/next/jump)
- [ ] Implement progress tracking (persist to ~/.wakawaka/progress.db)
- [ ] Implement reference mode (grammar index, reference cards, poem anthology)
- [ ] (Optional) Add live tutor toggle (disabled by default, uses `optional/live_tutor.py`)

### Verification
```bash
streamlit run app.py
# Manual check: navigate lessons, complete one, verify progress persists
# Verify no LLM calls during normal navigation (check logs)
```

---

## Phase 11: Integration Test
**Goal**: End-to-end validation with sample data.

### Tasks
- [ ] Create test corpus (3-5 poems)
- [ ] Run full pipeline
- [ ] Verify UI displays correctly
- [ ] Document any issues

### Full Pipeline Test
```bash
# Clean slate
rm -rf data/

# Run pipeline
python scripts/01_ingest_corpus.py --source test --output-dir data/raw/test
python scripts/02_annotate_corpus.py --input-dir data/raw --output data/annotated/poems.parquet
python scripts/03_extract_curriculum.py --input data/annotated/poems.parquet --output-dir data/curriculum
python scripts/04_generate_lessons.py --curriculum data/curriculum --poems data/annotated/poems.parquet --output-dir data/lessons --all
python scripts/05_compile_classroom.py --lessons data/lessons --curriculum data/curriculum --poems data/annotated/poems.parquet --output data/classroom.db

# Launch
streamlit run app.py
```

---

## Status Tracking

Update this section as phases complete:

| Phase | Status | Completed Date | Notes |
|-------|--------|----------------|-------|
| 0 | Complete | 2026-01-16 | Directory structure, requirements.txt, .gitignore, .env.example |
| 1 | Complete | 2026-01-16 | All Pydantic schemas created, 36 tests passing |
| 2 | Complete | 2026-01-16 | YAML prompts (annotate, generate_lesson, live_tutor) + prompt_loader.py |
| 3 | Complete | 2026-01-16 | 1000 poems (100 ogura100 + 900 lapis, native Japanese) ✓ |
| 4 | Complete | 2026-01-16 | 02_annotate_corpus.py; 1000 poems annotated with parallel processing |
| 5 | Complete | 2026-01-16 | 03_extract_curriculum.py; 1000 poems, 264 grammar points, 50 lessons |
| 5.5 | Complete | 2026-01-16 | NLP analysis scripts (lexical, collections, networks, phonetics, POS) |
| 5.5b | Complete | 2026-01-16 | 03b_refine_curriculum_llm.py; LLM redesigns to 7 thematic units |
| 6 | Complete | 2026-01-16 | 04_generate_lessons.py with checkpointing, caching, schema validation |
| 7 | Complete | 2026-01-16 | 05_compile_classroom.py; 50 lessons, 1000 poems, 264 grammar points compiled |
| 8 | Complete | 2026-01-16 | loader.py, progress.py, navigator.py; full curriculum navigation |
| 9 | Complete | 2026-01-16 | lesson.py, quiz.py, reference.py; interactive vocabulary hover |
| 10 | Complete | 2026-01-16 | app.py Streamlit; sidebar curriculum tree, lesson viewer, reference mode |
| 11 | Not Started | | |

---

## Cross-Conversation Continuity

When starting a new conversation:
1. Reference this file: `@SCAFFOLD_PLAN.md`
2. Check the Status Tracking table above
3. Continue from the next incomplete phase
4. Update status after completing each phase
5. Commit changes with phase number: `git commit -m "feat: complete phase N - [description]"`

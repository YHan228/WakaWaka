# WakaDecoder v2: Classical Japanese Poetry Classroom

## Build Specification for LLM-Assisted Development

---

## 0) Vision

Build a **self-contained interactive classroom** for learning classical Japanese poetry (e.g., waka/haiku) and the Japanese language needed, where:

- A **build pipeline** (offline, LLM-assisted) transforms raw corpora into structured lessons
- The **runtime interface** serves pre-generated content with optional live LLM calls upon user request (e.g., answer questions)
- The curriculum is **data-driven**: grammar points, difficulty, and lesson structure emerge from corpus analysis—not hardcoded
- The student is assumed to be a **Chinese native speaker** who can recognize hiragana and leverage kanji semantics from Chinese knowledge—no Japanese grammar knowledge, no Japanese vocabulary beyond kanji cognates

```
┌─────────────────────────────────────────────────────────────────┐
│                        BUILD PHASE                              │
│  Corpus → Annotation → Curriculum Extraction → Lesson Generation│
│                    (LLM-assisted, offline)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     classroom.json / SQLite
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       RUNTIME PHASE                             │
│         Student navigates pre-generated lessons                 │
│         (static serving, optional LLM for Q&A only)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1) Architecture Overview

### 1.1 Build Pipeline (Offline)

```
scripts/
├── 01_ingest_corpus.py       # Raw data collection (per RESOURCES.md)
├── 02_annotate_corpus.py     # LLM: extract grammar points, difficulty, readings
├── 03_extract_curriculum.py  # Derive lesson graph from annotation statistics
├── 04_generate_lessons.py    # LLM: generate teaching content per lesson
├── 05_compile_classroom.py   # Bundle into deployable format
```

### 1.2 Runtime Application

```
app.py                        # Streamlit UI (lesson navigator + viewer)
wakadecoder/
├── classroom/
│   ├── loader.py             # Load compiled classroom data
│   ├── progress.py           # Student progress tracking (local JSON/SQLite)
│   └── navigator.py          # Lesson sequencing logic
├── viewer/
│   ├── lesson.py             # Render lesson content
│   ├── quiz.py               # Render comprehension checks
│   └── reference.py          # Grammar reference cards
└── optional/
    └── live_tutor.py         # Optional: live LLM Q&A (disabled by default)
```

### 1.3 Data Flow

```
                    BUILD TIME                          RUNTIME
                    
[ONCOJ, Lapis, ...] 
        ↓
   01_ingest
        ↓
  data/raw/*.jsonl
        ↓
   02_annotate ←─── LLM (batch)
        ↓
  data/annotated/poems.parquet
    - text, reading, romaji
    - grammar_points[]        ← extracted, not predefined
    - difficulty_score        ← estimated
    - tokens[]                ← from Fugashi
        ↓
   03_extract_curriculum
        ↓
  data/curriculum/
    - grammar_index.json      ← discovered grammar points
    - lesson_graph.json       ← prerequisites, ordering
    - poem_assignments.json   ← which poems teach which points
        ↓
   04_generate_lessons ←─── LLM (batch)
        ↓
  data/lessons/
    - {lesson_id}.json        ← full lesson content
        ↓
   05_compile_classroom
        ↓
  data/classroom.db (or .json)  ──────────→  [app.py] → Student
```

---

## 2) Tech Stack

Install as you go. You are on Windows WSL Ubuntu. See `.env` for credentials (copy from `.env.example`).

### Build Phase
- Python 3.10+
- Fugashi + unidic-lite (tokenization)
- Pandas / Polars (data manipulation)
- **Gemini 3 Flash API** (only supported model; key in `.env`)
- google-generativeai (Gemini SDK)
- networkx (prerequisite graph cycle detection)
- python-dotenv (key management)
- SQLite or JSON (intermediate storage)

### Runtime Phase
- Streamlit (UI)
- SQLite or JSON (classroom data)
- No LLM dependency by default (optional Q&A module)

### To GitHub
You may commit at times you feel right. Repository URL and branch configured in `.env`.
If repo is not initialized yet, use values from `.env`:
```bash
git remote add origin $GITHUB_REPO_URL
git branch -M $GITHUB_BRANCH
git push -u origin $GITHUB_BRANCH
```
Coding agent related stuff and `.env` should be gitignored.

---

## 3) Build Pipeline Specifications

### 3.1 Script: `01_ingest_corpus.py`

**Purpose**: Collect raw materials from sources defined in `WakaDecoder_RESOURCES.md`.

**Output**: `data/raw/{source}/*.jsonl`

```json
{"id": "oncoj_001", "source": "oncoj", "text": "...", "author": "...", "collection": "...", "meta": {...}}
```

**Requirements**:
- Respect all rules in RESOURCES.md (rate limiting, ToS, provenance)
- CLI flags: `--source`, `--max-records`, `--output-dir`
- Idempotent: skip already-fetched records

---

### 3.2 Script: `02_annotate_corpus.py`

**Purpose**: Use LLM to annotate each poem with structured metadata.

**Input**: `data/raw/**/*.jsonl`

**Output**: `data/annotated/poems.parquet`

**Key Design Decisions**:
1. **Fugashi is source of truth for tokens** — LLM does not invent tokens, only provides readings
2. **Two-level grammar IDs** — `canonical_id` (e.g., "particle_ni") + `sense_id` (e.g., "location")
3. **Deterministic difficulty scoring** — computed from factors, not LLM-provided
4. **Span convention** — all spans are `[start, end)`, 0-based, end-exclusive (Python slice semantics)
5. **Text hash** — SHA256 prefix stored for integrity verification

**LLM Task** (per poem):

```yaml
Input:
  - poem text + text_hash (SHA256 prefix)
  - Fugashi tokenization with spans (SOURCE OF TRUTH)

Output (JSON):
  reading_hiragana: str
  reading_romaji: str
  token_readings:           # aligned to Fugashi tokens (enables deterministic ruby)
    - token_index: int
      reading_kana: str
  grammar_points:
    - canonical_id: str     # e.g., "particle_ni" (drives lessons)
      sense_id: str|null    # e.g., "location" (optional refinement)
      surface: str
      category: str
      description: str
      span: [int, int]      # 0-based, end-exclusive
  vocabulary:               # words Chinese speakers need to learn
    - word: str
      reading: str
      span: [int, int]
      meaning: str
      chinese_cognate_note: str|null
  difficulty_factors:
    - factor: str
      weight: float         # 0.0-1.0, from standardized table
      note: str|null
  semantic_notes: str
```

**Post-processing** (not LLM):
- `difficulty_score_computed` = deterministic formula from factors
- Validate all spans against text length
- Verify token_readings count matches Fugashi token count

**Key Design Principle**: `grammar_points` use **two-level identity**. The LLM extracts `canonical_id` + `sense_id`; curriculum is driven by `canonical_id` to avoid ID explosion.

**Requirements**:
- Batch processing with checkpointing (resume on failure)
- Cache LLM responses keyed by (poem_id, model, prompt_version)
- Validate output schema strictly; retry malformed responses once
- CLI flags: `--input-dir`, `--output`, `--model`, `--batch-size`, `--resume`

---

### 3.3 Script: `03_extract_curriculum.py`

**Purpose**: Derive lesson structure from annotation statistics—no hardcoding.

**Input**: `data/annotated/poems.parquet`

**Output**:
- `data/curriculum/grammar_index.json` — discovered grammar points (canonical + senses)
- `data/curriculum/lesson_graph.json` — lessons with prerequisites + cycle detection metadata
- `data/curriculum/prerequisite_graph.json` — raw prerequisite edges for analysis

**Key Design Decisions**:
1. **Two-level identity** — lessons driven by `canonical_id`; senses introduced progressively
2. **Prerequisite stoplist** — ultra-common particles (は, の, を, に, が) excluded from prerequisites
3. **Cycle detection** — uses networkx; breaks cycles by removing weakest edge
4. **Multi-criteria prerequisites** — requires co_ratio ≥70%, difficulty_gap ≥0.05, support ≥3 poems

**Algorithm**:

```python
# Step 1: Build grammar index (two-level)
# Group by canonical_id, aggregate senses within each

# Step 2: Compute co-occurrence matrix (canonical level only)
# Avoids ID explosion from sense combinations

# Step 3: Derive prerequisites (with safeguards)
for each canonical_id:
    for each co-occurring other_id:
        skip if other_id in STOPLIST  # ultra-common particles
        if co_ratio >= 0.7 AND difficulty_gap >= 0.05 AND support >= 3:
            add prerequisite edge (other_id → canonical_id)

# Step 4: Cycle detection
G = build_directed_graph(edges)
while cycles_exist(G):
    weakest_edge = find_weakest_in_cycle(G)  # by difficulty_gap, then co_ratio
    remove_edge(G, weakest_edge)
    log_removed_edge(weakest_edge)

# Step 5: Topological sort + group into units
topo_order = topological_sort(G)
group_by_category_into_units(topo_order)

# Step 6: Assign poems to lessons
# Each lesson gets N poems that contain the target canonical_id
# Sorted by difficulty (easier first)
```

**Output Schema** — `lesson_graph.json`:

```json
{
  "units": [
    {
      "id": "unit_particle",
      "title": null,
      "lessons": [
        {
          "id": "lesson_particle_wa",
          "canonical_grammar_point": "particle_wa",
          "senses_covered": ["topic", "contrast"],
          "prerequisites": [],
          "difficulty_tier": 1,
          "poem_ids": ["oncoj_042", "lapis_waka_117"]
        },
        {
          "id": "lesson_particle_ga",
          "canonical_grammar_point": "particle_ga",
          "senses_covered": ["subject", "emphatic"],
          "prerequisites": [],
          "difficulty_tier": 1,
          "poem_ids": ["oncoj_088", "lapis_waka_203"]
        }
      ]
    }
  ],
  "prerequisite_graph": {
    "edges": [
      {"from_id": "particle_mo", "to_id": "auxiliary_keri", "co_ratio": 0.75, "difficulty_gap": 0.12, "support_count": 8}
    ],
    "removed_edges": [],
    "stoplist_applied": ["particle_wa", "particle_no", "particle_wo", "particle_ni", "particle_ga"]
  },
  "meta": {
    "generated_at": "...",
    "corpus_size": 1500,
    "total_lessons": 47,
    "total_canonical_points": 35,
    "cycles_broken": 2,
    "stoplist_size": 5
  }
}
```

**Note**: `title` fields are null here—populated by LLM in step 03b or 04.

**Requirements**:
- Configurable: `--min-poems-per-lesson`, `--max-lessons`, `--difficulty-tiers`
- Output a report: `curriculum_report.md` summarizing discovered structure
- No hardcoded grammar lists

---

### 3.3b Script: `03b_refine_curriculum_llm.py` (Optional)

**Purpose**: Use LLM to reorganize algorithmic curriculum into pedagogically-sound thematic units.

**Input**: `data/curriculum/` (algorithmic curriculum from 03)

**Output**: `data/curriculum_refined/`
- `lesson_graph.json` — reorganized units with titles and prerequisites
- `llm_refinements.json` — raw LLM JSON response (or synthesis result)
- `curriculum_report.md` — human-readable summary
- `trials/` — (ensemble mode) individual trial proposals

**Key Design Decisions**:
1. **Non-destructive** — original curriculum preserved; refined output to separate directory
2. **Strict schema** — LLM outputs only valid JSON with exact lesson IDs provided
3. **Validation** — all lesson IDs verified; prerequisites must reference earlier units
4. **Ensemble mode** — run N independent trials, then synthesize best curriculum

**Ensemble Mode with Parallel Execution**:
```bash
python scripts/03b_refine_curriculum_llm.py --ensemble 10 --parallel 3  # 10 trials, 3 parallel
python scripts/03b_refine_curriculum_llm.py --ensemble 10 --parallel 5  # Faster with 5 parallel
```
- Each trial uses higher temperature (0.7) for diversity
- Synthesis uses lower temperature (0.3) for consistency
- Final curriculum cherry-picks best aspects from all trials
- Parallel execution speeds up trial generation 3-5x

**Prompts** (in `prompts/`):
- `curriculum_trial.yaml` — individual trial generation
- `curriculum_synthesis.yaml` — combine trials into final curriculum
- `lesson_context.yaml` — generate contextual guidance for lesson generation

**Lesson Context Generation**:
After synthesis, automatically generates `lesson_context.yaml` containing:
- Curriculum philosophy and progression logic
- Per-unit teaching emphasis
- Grammar-specific teaching notes
- Cross-cutting pedagogical guidelines

**Output Schema**:
```json
{
  "units": [
    {"id": "unit_01", "title": "Fundamental Particles", "lessons": ["lesson_particle_no", ...]}
  ],
  "prerequisites": {
    "lesson_auxiliary_zu": ["lesson_conjugation_mizenkei"]
  }
}
```

**Requirements**:
- Uses high-quality model (gemini-3-pro-preview) for pedagogical reasoning
- Validates all lesson IDs exist in original curriculum
- Validates prerequisites reference lessons in earlier units only
- CLI: `--input-dir`, `--output-dir`, `--model`, `--ensemble N`

---

### 3.4 Script: `04_generate_lessons.py`

**Purpose**: Generate complete lesson content using LLM.

**Two-Step Generation Process**:
1. **Poem Selection** (optional, via `--select-poems`): LLM selects 2-3 best teaching poems from candidate pool of 10
2. **Lesson Generation**: LLM generates full lesson content using selected poems

**Input**:
- `data/curriculum_refined/lesson_graph.json` (preferred) or `data/curriculum/lesson_graph.json`
- `data/annotated/poems.parquet`

**Curriculum Structure** (from Phase 5):
Each lesson contains:
- `candidate_poem_ids` — pool of 10 poems (algorithmically selected by difficulty)
- `poem_ids` — final selected poems (empty initially, populated by LLM selection step)

**Output**: `data/lessons/{lesson_id}.json`

**Prompts** (in `prompts/`):
- `select_poems.yaml` — LLM poem selection from candidates
- `generate_lesson.yaml` — full lesson content generation

**LLM Task** (per lesson):

```yaml
Context provided to LLM:
  - grammar_point metadata (from grammar_index)
  - assigned poems with full annotations
  - prerequisite lessons (titles + summaries only)
  - student level assumption: "Chinese native speaker; knows hiragana, leverages kanji from Chinese, no Japanese grammar"

LLM generates:
  lesson_title: str               # human-readable
  lesson_summary: str             # 1-2 sentences
  grammar_explanation:
    concept: str                  # what this grammar point does
    formation: str                # how it's formed (if applicable)
    variations: [str]             # related forms
    common_confusions: [str]      # vs. similar patterns
    logic_analogy: str            # optional: explain like a data structure/function
  
  teaching_sequence:              # ordered list of teaching steps
    - type: "introduction"
      content: str
    - type: "poem_presentation"
      poem_id: str
      display:
        text_with_furigana: str   # HTML or markdown with ruby
        romaji: str
        translation: str          # English translation
      vocabulary:                 # key words for Chinese speakers
        - word: str               # Japanese word
          reading: str            # hiragana
          meaning: str            # meaning (may note Chinese cognate)
      focus_highlight: [int, int] # span of target grammar point
    - type: "grammar_spotlight"
      content: str                # explain the grammar point IN THIS POEM
      evidence: str               # reference specific tokens
    - type: "contrast_example"    # optional: show how meaning changes
      content: str
    - type: "poem_presentation"
      poem_id: str
      display: {...}              # same structure as above
      vocabulary: [...]           # same structure as above
    - type: "comprehension_check"
      question: str
      answer: str
      hint: str
    - type: "summary"
      content: str
  
  reference_card:                 # quick reference for review
    point: str
    one_liner: str
    example: str
    see_also: [str]               # related lesson IDs
  
  forward_references:             # grammar in poems not yet taught
    - point: str
      note: str                   # e.g., "We'll learn けり in Unit 3"
```

**Requirements**:
- Batch with checkpointing
- Validate output schema
- CLI: `--lesson-ids` (specific lessons), `--all`, `--model`, `--resume`, `--select-poems`
- Generate `lessons_manifest.json` listing all generated lessons

**Example Usage**:
```bash
# Full pipeline with parallel generation (recommended)
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --all --select-poems --parallel 3 --resume

# Sequential generation (safer, easier to debug)
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --all --select-poems --resume

# Without poem selection (uses first 3 candidates)
python scripts/04_generate_lessons.py --curriculum data/curriculum_refined --all --resume
```

**Lesson Context Integration**:
If `lesson_context.yaml` exists in the curriculum directory, it is automatically loaded and injected into the system prompt for consistent pedagogy across all lessons.

---

### 3.5 Script: `05_compile_classroom.py`

**Purpose**: Bundle all generated content into deployable format.

**Input**: `data/lessons/*.json`, `data/curriculum/*.json`, `data/annotated/poems.parquet`

**Output**: `data/classroom.db` (SQLite) or `data/classroom/` (JSON files)

**Schema** (SQLite):

```sql
CREATE TABLE units (
    id TEXT PRIMARY KEY,
    title TEXT,
    position INTEGER
);

CREATE TABLE lessons (
    id TEXT PRIMARY KEY,
    unit_id TEXT REFERENCES units(id),
    title TEXT,
    summary TEXT,
    position INTEGER,
    content JSON,          -- full lesson JSON
    prerequisites JSON     -- array of lesson IDs
);

CREATE TABLE poems (
    id TEXT PRIMARY KEY,
    text TEXT,
    reading_hiragana TEXT,
    reading_romaji TEXT,
    annotations JSON
);

CREATE TABLE grammar_points (
    id TEXT PRIMARY KEY,
    category TEXT,
    summary TEXT,
    reference_card JSON,
    lesson_id TEXT REFERENCES lessons(id)
);

-- Note: progress table is stored separately in ~/.wakadecoder/progress.db
-- NOT in classroom.db (which is read-only content)
CREATE TABLE progress (
    lesson_id TEXT PRIMARY KEY,
    status TEXT,           -- "not_started", "in_progress", "completed"
    completed_at TEXT,
    quiz_score REAL
);
```

**Requirements**:
- Integrity check: all referenced poem_ids exist, all prerequisites exist
- Generate `classroom_stats.json`: total lessons, poems, estimated study hours
- Optionally minify JSON for smaller payload

---

## 4) Runtime Application Specifications

### 4.1 `app.py` — Main Interface

**Layout**:

**Features**: Romaji annotation, vocabulary (with Chinese cognate notes), and English translation are included in poem presentations.

```
┌──────────────────────────────────────────────────────────────────┐
│  WakaDecoder Classroom                         [Progress: 12/47] │
├────────────────┬─────────────────────────────────────────────────┤
│ CURRICULUM     │                                                 │
│                │  Lesson 5: The Particle が                      │
│ ▼ Unit 1       │  ─────────────────────────────────────────────  │
│   ✓ は         │                                                 │
│   → が         │  [Introduction content rendered here]           │
│   ○ を         │                                                 │
│   ○ に         │  ┌─────────────────────────────────────────┐   │
│                │  │  秋来ぬと目にはさやかに見えねども       │   │
│ ▶ Unit 2       │  │  あきこぬと めにはさやかに みえねども   │   │
│                │  │                                         │   │
│ ▶ Unit 3       │  │  [が is not in this poem—contrast!]     │   │
│                │  └─────────────────────────────────────────┘   │
│                │                                                 │
│ ──────────     │  GRAMMAR SPOTLIGHT:                             │
│ [Reference]    │  Notice this poem uses は (which we learned)... │
│ [Settings]     │                                                 │
│                │  ┌─────────────────────────────────────────┐   │
│                │  │ ✎ Comprehension Check                   │   │
│                │  │ Q: Why does the poet use は here?       │   │
│                │  │ [Show Answer] [Need Hint]               │   │
│                │  └─────────────────────────────────────────┘   │
│                │                                                 │
│                │  [← Previous] [Mark Complete] [Next →]          │
└────────────────┴─────────────────────────────────────────────────┘
```

**Requirements**:
- Sidebar: curriculum tree with progress indicators
- Main area: lesson content renderer (markdown/HTML)
- Navigation: previous/next, jump to any unlocked lesson
- Progress persists locally in `~/.wakadecoder/progress.db` (SQLite, separate from classroom content)
- No LLM calls during normal lesson navigation

### 4.2 Reference Mode

Accessible from sidebar:
- Grammar point index (searchable)
- Reference cards for all learned points
- Poem anthology (browse all poems, grouped by collection)

### 4.3 Optional: Live Tutor (`optional/live_tutor.py`)

**Disabled by default**. When enabled:
- "Ask a question" button appears in lesson view
- User types question about current lesson/poem
- LLM responds with context: current lesson + poem + student progress

**Prompt structure**:

```
CONTEXT:
- Student is on: {lesson_title}
- Grammar point being taught: {grammar_point}
- Current poem: {poem_text}
- Student has completed: {completed_lessons}
- Student has NOT learned: {future_lessons}

RULES:
- Answer only using concepts already taught
- If question requires future knowledge, say "We'll cover that in {lesson_name}"
- Assume student is Chinese native speaker who knows hiragana and leverages kanji from Chinese

STUDENT QUESTION:
{user_question}
```

---

## 5) Deliverables Summary

### Directory Structure

```
wakadecoder/
├── app.py
├── wakadecoder/
│   ├── classroom/
│   │   ├── loader.py
│   │   ├── progress.py
│   │   └── navigator.py
│   ├── viewer/
│   │   ├── lesson.py
│   │   ├── quiz.py
│   │   └── reference.py
│   └── optional/
│       └── live_tutor.py
├── scripts/
│   ├── 01_ingest_corpus.py
│   ├── 02_annotate_corpus.py
│   ├── 03_extract_curriculum.py
│   ├── 04_generate_lessons.py
│   └── 05_compile_classroom.py
├── data/                        # .gitignore'd
│   ├── raw/
│   ├── annotated/
│   ├── curriculum/
│   ├── lessons/
│   └── classroom.db
├── prompts/                     # LLM prompt templates
│   ├── annotate.yaml
│   ├── generate_lesson.yaml
│   └── live_tutor.yaml
├── tests/
│   ├── test_annotation_schema.py
│   ├── test_lesson_schema.py
│   └── test_classroom_loader.py
├── requirements.txt
├── .env.example
├── README.md
└── WakaDecoder_RESOURCES.md     # unchanged
```

### Files to Provide

| File | Purpose |
|------|---------|
| `requirements.txt` | All dependencies |
| `.env.example` | `GEMINI_API_KEY`, `SUDO_PASSWORD`, `GITHUB_REPO_URL` |
| `README.md` | Setup, build pipeline usage, runtime usage |
| `prompts/*.yaml` | Editable prompt templates (not embedded in code) |

---

## 6) Acceptance Criteria

### Build Pipeline
- [ ] `01_ingest` successfully fetches from at least one source (ONCOJ recommended for MVP)
- [ ] `02_annotate` produces valid `poems.parquet` with schema-conformant annotations
- [ ] `03_extract_curriculum` generates `lesson_graph.json` with ≥10 lessons from corpus
- [ ] `04_generate_lessons` produces valid lesson JSON for all lessons in graph
- [ ] `05_compile_classroom` produces loadable `classroom.db`

### Runtime
- [ ] `streamlit run app.py` loads classroom and displays curriculum tree
- [ ] Lesson navigation works (previous/next/jump)
- [ ] Progress persists across sessions
- [ ] Reference mode shows grammar index
- [ ] No LLM calls occur during normal lesson navigation (verify via logging)

### Quality
- [ ] No hardcoded grammar points in curriculum extraction
- [ ] Prompts are in external YAML files, not embedded strings
- [ ] All scripts have `--help` with documented flags
- [ ] Tests pass for annotation schema, lesson schema, and classroom loader

---

## 7) Implementation Notes

### MVP Scope Reduction

For initial prototype, acceptable simplifications:
- Single source (ONCOJ only)
- Corpus size: 100-300 poems
- Lesson count: 10-20
- Skip live tutor module
- JSON files instead of SQLite

### Prompt Engineering Priority

The quality of the classroom depends heavily on:
1. `annotate.yaml` — accurate grammar point extraction
2. `generate_lesson.yaml` — pedagogically sound lesson content

Iterate on these prompts with sample poems before running full pipeline.

### Estimated Build Costs (100 poems, GPT-4o)

| Step | Calls | Est. Tokens/Call | Cost |
|------|-------|------------------|------|
| 02_annotate | 100 | ~2000 | ~$0.60 |
| 04_generate_lessons | ~15 | ~4000 | ~$0.36 |
| **Total** | | | **~$1.00** |

Scaling to 1000 poems: ~$10-15 total build cost.

---

## 8) References

- Corpus acquisition: see `WakaDecoder_RESOURCES.md`
- Fugashi documentation: https://github.com/polm/fugashi
- UniDic: https://clrd.ninjal.ac.jp/unidic/

---

## Appendix A: Schema Definitions

See `wakadecoder/schemas/` for Pydantic models (to be generated).

## Appendix B: Sample Prompt Templates

See `prompts/*.yaml` (to be generated).

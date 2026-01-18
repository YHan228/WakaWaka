# WakaWaka

**Classical Japanese Poetry Learning Platform for Chinese Speakers**

WakaWaka teaches classical Japanese poetry (waka/和歌) to Chinese-speaking learners by leveraging shared kanji knowledge. The platform uses a sophisticated build pipeline with LLM-assisted content generation, producing a self-contained web app that requires no API calls at runtime.

## Quick Start

### For Users

Visit: **[wakawaka.streamlit.app](https://wakawaka.streamlit.app)** *(coming soon)*

Or run locally:
```bash
git clone https://github.com/YHan228/WakaWaka.git
cd WakaWaka
pip install -r requirements.txt
streamlit run app.py
```

### For Developers

```bash
# Set up environment
cp .env.example .env  # Add your GEMINI_API_KEY
pip install -r requirements.txt

# Run the full build pipeline
python scripts/01_ingest_corpus.py --source all
python scripts/02_annotate_corpus.py
python scripts/03_extract_curriculum.py
python scripts/03b_refine_curriculum_llm.py --ensemble 10
python scripts/03c_select_poems.py --sequential
python scripts/04_generate_lessons.py --all --parallel 10
python scripts/05_compile_classroom.py
python scripts/06_annotate_literary.py
python scripts/07_generate_audio.py --parallel 100

# Launch the app
streamlit run app.py
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUILD PIPELINE (Offline, LLM-Assisted)              │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Ingest  │──▶│ Annotate │──▶│ Extract  │──▶│  Refine  │──▶│  Select  │  │
│  │  Corpus  │   │   NLP    │   │Curriculum│   │   LLM    │   │  Poems   │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │              │              │              │              │         │
│       ▼              ▼              ▼              ▼              ▼         │
│   poems.json    poems.parquet  lesson_graph   refined_graph  with_poems    │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │ Generate │──▶│ Compile  │──▶│ Literary │──▶│  Audio   │                 │
│  │ Lessons  │   │ Database │   │ Analysis │   │   TTS    │                 │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘                 │
│       │              │              │              │                        │
│       ▼              ▼              ▼              ▼                        │
│  lessons/*.json classroom.db  literary.parquet audio/*.mp3                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME (Streamlit, No LLM)                         │
│                                                                             │
│   classroom.db ──▶ Loader ──▶ Navigator ──▶ Viewer ──▶ UI                  │
│   literary.parquet ─────────────────────────┘    └── Audio Player          │
│   audio/*.mp3 ──────────────────────────────────────┘                      │
│                                                                             │
│   Features: Lessons • Quizzes • Reference Cards • Poem Anthology • Audio   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Build Pipeline Scripts

| Step | Script | Input | Output | Description |
|------|--------|-------|--------|-------------|
| 1 | `01_ingest_corpus.py` | Web sources | `poems.json` | Scrape poems from Ogura 100 and Lapis collections |
| 2 | `02_annotate_corpus.py` | `poems.json` | `poems.parquet` | Morphological analysis with Fugashi + UniDic-waka |
| 3 | `03_extract_curriculum.py` | `poems.parquet` | `lesson_graph.json` | Extract grammar points, build prerequisite DAG |
| 3b | `03b_refine_curriculum_llm.py` | `lesson_graph.json` | `refined_graph.json` | Ensemble LLM optimization (10 trials + synthesis) |
| 3c | `03c_select_poems.py` | `refined_graph.json` | `with_poems.json` | LLM-assisted poem selection with reuse monitoring |
| 4 | `04_generate_lessons.py` | `with_poems.json` | `lessons/*.json` | Generate lesson content with Gemini (parallel) |
| 5 | `05_compile_classroom.py` | All artifacts | `classroom.db` | Bundle everything into SQLite for runtime |
| 6 | `06_annotate_literary.py` | `poems.parquet` | `literary.parquet` | Literary analysis: devices, imagery, themes |
| 7 | `07_generate_audio.py` | `classroom.db` | `audio/*.mp3` | TTS pronunciation with Google Cloud (parallel) |

### Key Technologies

**NLP & Tokenization**
- **Fugashi** (MeCab wrapper) for morphological analysis
- **UniDic-waka** custom dictionary for classical Japanese poetry
- Token-level readings, POS tags, and lemma extraction

**LLM Integration (Build-time only)**
- **Google Gemini** (Flash/Pro) for:
  - Curriculum refinement (ensemble trials + judge synthesis)
  - Poem selection for lessons
  - Lesson content generation
  - Literary analysis

**Audio**
- **Google Cloud TTS** with Neural2/Wavenet voices
- Poetic SSML structure (kami-no-ku pause + shimo-no-ku)
- Random voice selection for variety

**Runtime**
- **Streamlit** web framework
- **SQLite** for data serving (no external dependencies)
- All content pre-generated (zero LLM calls at runtime)

---

## Data Files

After running the build pipeline:

```
data/
├── classroom.db              # Main runtime database (lessons, poems, grammar)
├── classroom.stats.json      # Build statistics
├── annotated/
│   └── poems.parquet         # Annotated poems with morphology
├── curriculum_refined/
│   ├── lesson_graph.json     # Original curriculum DAG
│   └── lesson_graph_with_poems.json  # With selected poems
├── lessons/
│   ├── lesson_*.json         # Individual lesson content files
│   └── lessons_manifest.json # Lesson index
├── literary/
│   └── poems_literary.parquet  # Literary analysis data
├── audio/
│   └── *.mp3                 # Poem pronunciation audio (1000 files)
└── introduction.json         # Course introduction content
```

---

## Curriculum Structure

### Grammar Progression

The curriculum follows a prerequisite-based DAG:

```
Unit 1: Particles (Foundation)
├── の (genitive) → は (topic) → が (subject) → を (object)
├── に (location) → で (means) → と (quotation)
└── も (also) → や (or)

Unit 2: Conjugations
├── 連用形 (continuative) → 連体形 (attributive)
└── 已然形 (realis) → 命令形 (imperative)

Unit 3: Auxiliaries
├── なり (copula) → けり (past exclamatory)
├── つ/ぬ (perfective) → たり/り (resultative)
└── む (volitional) → べし (necessity)

Unit 4-7: Advanced (系結び, 切字, 修辞技法...)
```

### Lesson Content

Each lesson includes:
- **Grammar explanation** with formation patterns
- **Poem presentations** with interactive vocabulary
- **Grammar spotlights** showing the point in context
- **Literary insights** connecting to Chinese poetry traditions
- **Comprehension checks** with self-assessment
- **Reference cards** for quick review

---

## Features

### Interactive Vocabulary

Poems display with color-coded word types:
- **Blue**: Particles (の, は, を, に...)
- **Green**: Verbs
- **Purple**: Auxiliaries (けり, なり, む...)
- **Orange**: Adjectives
- **Black**: Nouns

Hover/tap reveals reading, meaning, and Chinese cognate notes.

### Literary Analysis

Each poem includes expandable literary analysis:
- Interpretation and emotional tone
- Poetic devices (掛詞, 縁語, 枕詞...)
- Seasonal context (四季)
- Chinese poetry parallels

### Audio Playback

TTS audio for all 1000 poems:
- Natural Japanese pronunciation
- Poetic pacing with verse pauses
- Multiple voice varieties

### Progress Tracking

- Lesson completion status
- Prerequisite-based unlocking
- Reference card collection

---

## Development

### Project Structure

```
wakadecoder/
├── app.py                    # Streamlit entry point
├── wakawaka/
│   ├── classroom/            # Runtime: loader, progress, navigator
│   ├── viewer/               # Rendering: lessons, quizzes, literary
│   ├── schemas/              # Pydantic models
│   └── utils/                # Helpers
├── scripts/                  # Build pipeline
├── prompts/                  # LLM prompt templates (YAML)
├── data/                     # Generated data (gitignored)
└── tests/                    # Validation tests
```

### Testing Checkpoints

After each build step:
```bash
# Step 2: Annotation
python -c "import pandas as pd; df = pd.read_parquet('data/annotated/poems.parquet'); print(f'{len(df)} poems annotated')"

# Step 3-3c: Curriculum
python -c "import json; g = json.load(open('data/curriculum_refined/lesson_graph_with_poems.json')); print(f\"{len(g['units'])} units, {sum(len(u['lessons']) for u in g['units'])} lessons\")"

# Step 4: Lessons
ls data/lessons/lesson_*.json | wc -l

# Step 5: Database
sqlite3 data/classroom.db "SELECT COUNT(*) FROM lessons"

# Full test
streamlit run app.py
```

### Git Workflow

- Never commit `.env`, `data/`, `__pycache__/`
- Use conventional commits: `feat:`, `fix:`, `docs:`
- Commit after completing each pipeline phase

---

## Configuration

### Environment Variables

```bash
# Required for build pipeline
GEMINI_API_KEY=your_key_here

# Optional for audio generation
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Build Options

```bash
# Curriculum refinement
python scripts/03b_refine_curriculum_llm.py \
    --ensemble 10 \           # Number of trial generations
    --model gemini-3-pro-preview

# Lesson generation
python scripts/04_generate_lessons.py \
    --all \                   # Generate all lessons
    --parallel 10 \           # Parallel workers
    --model gemini-3-pro-preview

# Audio generation
python scripts/07_generate_audio.py \
    --parallel 100 \          # Parallel TTS requests
    --speaking-rate 0.85      # Slower for learners
```

---

## License

MIT

## Acknowledgments

- Poem corpus from public domain classical Japanese collections
- Morphological analysis powered by MeCab/UniDic
- LLM capabilities provided by Google Gemini
- TTS powered by Google Cloud Text-to-Speech

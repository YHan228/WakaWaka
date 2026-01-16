# WakaWaka

**Classical Japanese Poetry for Chinese Speakers**

WakaWaka is an interactive learning platform that teaches classical Japanese poetry (waka/和歌) to Chinese-speaking learners. It leverages your existing kanji knowledge to decode 1000+ years of Japanese poetic tradition.

## For Users

### What is this?

A web app that teaches you to read classical Japanese poetry (waka) by treating your Chinese character knowledge as a superpower:

- **You already know 70% of the meaning** — kanji in waka share roots with Chinese characters
- **Learn the "operators"** — hiragana particles (の、は、を) are the code that connects known concepts
- **Poetry as data** — each poem is a structured dataset where kanji are objects, hiragana are functions

### Who is it for?

- Native Chinese speakers who can read hanzi fluently
- Know basic hiragana
- No prior Japanese grammar knowledge required
- Interested in classical literature and East Asian aesthetics

### How to use

Visit: **[wakawaka.streamlit.app](https://wakawaka.streamlit.app)** *(coming soon)*

Or run locally:
```bash
git clone https://github.com/YHan228/WakaWaka.git
cd WakaWaka
pip install -r requirements.txt
streamlit run app.py
```

---

## Methodology

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BUILD PIPELINE (Offline)                     │
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │  Corpus  │───▶│ Annotate │───▶│Curriculum│───▶│ Generate │      │
│   │  Ingest  │    │   NLP    │    │ Extract  │    │ Lessons  │      │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│        │               │               │               │             │
│        ▼               ▼               ▼               ▼             │
│   poems.json      poems.parquet   lesson_graph    lessons/*.json    │
│   (raw text)      (+ morphology)  (DAG structure) (teaching content)│
│                                                                      │
│   ┌──────────┐    ┌──────────────┐                                  │
│   │ Compile  │───▶│ classroom.db │ ◀── Single SQLite for runtime    │
│   │ Database │    └──────────────┘                                  │
│   └──────────┘                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RUNTIME (Streamlit)                          │
│                                                                      │
│   classroom.db ──▶ Loader ──▶ Navigator ──▶ Viewer ──▶ UI           │
│                                                                      │
│   No LLM calls at runtime — all content pre-generated               │
└─────────────────────────────────────────────────────────────────────┘
```

### Build Pipeline

| Step | Script | Input | Output | Method |
|------|--------|-------|--------|--------|
| 1 | `01_ingest_corpus.py` | Online sources | `poems.json` | Web scraping + text cleaning |
| 2 | `02_annotate_corpus.py` | `poems.json` | `poems.parquet` | Fugashi (MeCab) morphological analysis |
| 3 | `03_extract_curriculum.py` | `poems.parquet` | `lesson_graph.json` | Grammar extraction + DAG construction |
| 3b | `03b_refine_curriculum_llm.py` | `lesson_graph.json` | Refined graph | LLM curriculum optimization |
| 4 | `04_generate_lessons.py` | Graph + poems | `lessons/*.json` | LLM lesson generation (parallel) |
| 5 | `05_compile_classroom.py` | All artifacts | `classroom.db` | SQLite compilation |
| 6 | `06_generate_introduction.py` | Curriculum context | `introduction.json` | Ensemble LLM + judge synthesis |

### Key Techniques

#### 1. Morphological Analysis (Step 2)
```
Input:  "秋の田の"
         ↓ Fugashi/MeCab + UniDic
Output: [秋/名詞, の/助詞, 田/名詞, の/助詞]
        + readings, lemmas, POS tags
```
- **Fugashi** is the source of truth for tokenization
- LLM never invents tokens — only explains them

#### 2. Curriculum Graph (Step 3)
```
           ┌─────────────┐
           │ Particle の │ (Lesson 1)
           └──────┬──────┘
                  │ prerequisite
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ Particle は   │   │ Particle を   │
└───────────────┘   └───────────────┘
```
- Grammar points extracted from corpus frequency
- Dependencies form a DAG (no cycles)
- Topological sort determines lesson order

#### 3. LLM Lesson Generation (Step 4)
```
┌────────────────────────────────────────────────┐
│              For each lesson:                   │
│                                                 │
│  Context = {                                    │
│    grammar_point,                               │
│    prerequisite_summaries,                      │
│    selected_poems (via LLM ranking),            │
│    annotated_tokens                             │
│  }                                              │
│           │                                     │
│           ▼                                     │
│  ┌─────────────────┐                           │
│  │   Gemini API    │  structured JSON output    │
│  │  (Flash model)  │ ──────────────────────▶   │
│  └─────────────────┘                           │
│           │                                     │
│           ▼                                     │
│  Pydantic validation + retry on failure         │
└────────────────────────────────────────────────┘
```

#### 4. Ensemble Introduction Generation (Step 6)
```
┌─────────────────────────────────────────────────────┐
│                                                      │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│   │ Trial 1 │  │ Trial 2 │  │ Trial 3 │  ...       │
│   │ (T=0.9) │  │ (T=0.9) │  │ (T=0.9) │            │
│   └────┬────┘  └────┬────┘  └────┬────┘            │
│        │            │            │                  │
│        └────────────┼────────────┘                  │
│                     ▼                               │
│              ┌────────────┐                         │
│              │   Judge    │  T=0.3 (deterministic)  │
│              │ Synthesize │                         │
│              └─────┬──────┘                         │
│                    ▼                                │
│           Best elements combined                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```
- Multiple high-temperature trials for diversity
- Low-temperature judge selects best elements
- Produces more robust, well-rounded content

### Runtime Architecture

```
wakawaka/
├── classroom/
│   ├── loader.py      # SQLite → Pydantic models
│   ├── progress.py    # User progress tracking
│   └── navigator.py   # Lesson sequencing + prerequisites
├── viewer/
│   ├── lesson.py      # Interactive poem rendering
│   ├── quiz.py        # Comprehension checks
│   └── reference.py   # Grammar reference cards
└── schemas/           # Pydantic models for all data types
```

**Interactive Vocabulary Display:**
```
┌─────────────────────────────────────────┐
│  秋  の   田  の   庵  の   苫  を      │
│  ▲       ▲       ▲       ▲             │
│  │       │       │       │             │
│ noun  particle noun  particle          │
│(autumn)  (の)  (field) (の)            │
│                                         │
│  [Hover over any word for details]      │
└─────────────────────────────────────────┘
```
- Color-coded by word type (particles=blue, verbs=green, etc.)
- Hover/tap reveals: reading + meaning + Chinese cognate notes

---

## Tech Stack

- **Runtime**: Streamlit, SQLite
- **NLP**: Fugashi (MeCab wrapper), UniDic
- **LLM**: Google Gemini API (build-time only)
- **Data**: Pydantic models, Parquet, JSON

## License

MIT

## Acknowledgments

- Poem corpus from public domain classical Japanese collections
- Morphological analysis powered by MeCab/UniDic

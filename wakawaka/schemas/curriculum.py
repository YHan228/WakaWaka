"""
Curriculum schemas for WakaDecoder.

Defines Pydantic models for curriculum structure including:
- Grammar index (two-level: canonical + senses)
- Prerequisite graph with cycle detection metadata
- Lesson graph with units and lessons
"""

from pydantic import BaseModel, Field
from typing import Optional

# -----------------------------------------------------------------------------
# Grammar index (two-level: canonical + senses)
# -----------------------------------------------------------------------------


class SenseEntry(BaseModel):
    """A specific sense/usage of a canonical grammar point."""
    sense_id: str
    surfaces: list[str]      # observed surface forms for this sense
    frequency: int           # poem count with this sense
    avg_difficulty: float
    example_poem_ids: list[str] = []  # up to 3 examples


class GrammarIndexEntry(BaseModel):
    """
    Index entry for a canonical grammar point.
    Lessons are driven by canonical_id; senses are introduced progressively.
    """
    canonical_id: str
    category: str
    surfaces: list[str]      # all observed surface forms (across all senses)
    frequency: int           # total poem count
    avg_difficulty: float    # weighted average across senses
    senses: list[SenseEntry] = []  # breakdown by sense
    co_occurrences: dict[str, int] = {}  # other canonical_id -> count


class GrammarIndex(BaseModel):
    entries: dict[str, GrammarIndexEntry]  # keyed by canonical_id
    generated_at: str
    corpus_size: int


# -----------------------------------------------------------------------------
# Prerequisite graph with cycle detection
# -----------------------------------------------------------------------------

# Ultra-common points that should NOT be prerequisites of everything
# These are "implicitly taught" in early lessons
STOPLIST_CANONICAL_IDS = {
    'particle_wa',   # topic marker — appears everywhere
    'particle_no',   # genitive — appears everywhere
    'particle_wo',   # object marker — very common
    'particle_ni',   # very common (multiple senses)
    'particle_ga',   # subject marker — very common
}


class PrerequisiteEdge(BaseModel):
    """An edge in the prerequisite graph."""
    from_id: str         # prerequisite (must be learned first)
    to_id: str           # dependent (requires prerequisite)
    co_ratio: float      # what fraction of to_id poems contain from_id
    difficulty_gap: float  # to_id.difficulty - from_id.difficulty
    support_count: int   # absolute co-occurrence count


class PrerequisiteGraph(BaseModel):
    """
    Prerequisite relationships between canonical grammar points.
    Includes cycle detection metadata.
    """
    edges: list[PrerequisiteEdge]
    removed_edges: list[PrerequisiteEdge] = []  # edges removed to break cycles
    stoplist_applied: list[str] = []  # canonical_ids excluded from prereqs


# -----------------------------------------------------------------------------
# Lesson graph
# -----------------------------------------------------------------------------


class LessonNode(BaseModel):
    id: str
    canonical_grammar_point: str  # canonical_id (lessons teach canonical concepts)
    senses_covered: list[str] = []  # which senses this lesson introduces
    prerequisites: list[str]  # lesson IDs
    difficulty_tier: int = Field(..., ge=1, le=5)  # 1=easiest, 5=hardest
    candidate_poem_ids: list[str]  # larger pool for LLM to select from (10 poems)
    poem_ids: list[str] = []  # final selected poems (populated by LLM selection step)


class Unit(BaseModel):
    id: str
    title: Optional[str] = None  # populated by lesson generation
    lessons: list[LessonNode]


class LessonGraph(BaseModel):
    units: list[Unit]
    prerequisite_graph: PrerequisiteGraph
    meta: dict  # generated_at, corpus_size, total_lessons, etc.

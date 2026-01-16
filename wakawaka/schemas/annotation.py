"""
Annotation schemas for WakaDecoder.

Defines Pydantic models for poem annotation data including:
- Fugashi tokenization (source of truth for tokens)
- LLM-provided readings
- Grammar points with two-level identity (canonical_id + sense_id)
- Vocabulary for Chinese speakers
- Difficulty scoring
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import Optional
import hashlib

# =============================================================================
# SPAN CONVENTION: All spans are [start, end) — 0-based, end-exclusive
# This matches Python slice semantics: text[start:end]
# =============================================================================


def validate_span(v: list[int]) -> list[int]:
    """Shared span validation: 0-based, end-exclusive."""
    if len(v) != 2 or v[0] < 0 or v[1] <= v[0]:
        raise ValueError('Invalid span: must be [start, end) where 0 <= start < end')
    return v


# -----------------------------------------------------------------------------
# Token-related schemas (Fugashi is source of truth)
# -----------------------------------------------------------------------------

class FugashiToken(BaseModel):
    """Token from Fugashi/UniDic — this is the SOURCE OF TRUTH for tokenization."""
    surface: str
    pos: str              # part of speech
    pos_detail: str       # detailed POS (e.g., "助詞,格助詞,一般")
    lemma: str
    span: list[int] = Field(..., min_length=2, max_length=2)  # char offsets in poem text

    @field_validator('span')
    @classmethod
    def span_valid(cls, v):
        return validate_span(v)


class TokenReading(BaseModel):
    """LLM-provided reading for a Fugashi token (enables deterministic ruby)."""
    token_index: int = Field(..., ge=0)  # index into fugashi_tokens array
    reading_kana: str                     # hiragana reading for this token


# -----------------------------------------------------------------------------
# Grammar point schema (two-level identity)
# -----------------------------------------------------------------------------

class GrammarPoint(BaseModel):
    """
    Grammar point with two-level identity:
    - canonical_id: main grammar concept (used for curriculum/lessons)
    - sense_id: specific usage variant (optional, for detailed analysis)

    Example: canonical_id="particle_ni", sense_id="location"
    """
    canonical_id: str = Field(..., pattern=r'^[a-z]+_[a-z0-9_]+$')
    sense_id: Optional[str] = Field(default=None, pattern=r'^[a-z0-9_]+$')
    surface: str
    category: str = Field(..., pattern=r'^(particle|auxiliary|conjugation|kireji|syntax|other)$')
    description: str
    span: list[int] = Field(..., min_length=2, max_length=2)

    @field_validator('span')
    @classmethod
    def span_valid(cls, v):
        return validate_span(v)

    @computed_field
    @property
    def full_id(self) -> str:
        """Full ID combining canonical and sense (for backward compatibility)."""
        if self.sense_id:
            return f"{self.canonical_id}_{self.sense_id}"
        return self.canonical_id


# -----------------------------------------------------------------------------
# Vocabulary schema (for Chinese speakers)
# -----------------------------------------------------------------------------

class VocabularyAnnotation(BaseModel):
    """Vocabulary item extracted during annotation — words Chinese speakers need to learn."""
    word: str
    reading: str                          # hiragana
    span: list[int] = Field(..., min_length=2, max_length=2)
    meaning: str
    chinese_cognate_note: Optional[str] = None  # explains relation to Chinese

    @field_validator('span')
    @classmethod
    def span_valid(cls, v):
        return validate_span(v)


# -----------------------------------------------------------------------------
# Difficulty scoring (deterministic from factors)
# -----------------------------------------------------------------------------

# Canonical difficulty factor weights (used for deterministic scoring)
DIFFICULTY_WEIGHTS = {
    'classical_auxiliary': 0.25,
    'archaic_particle': 0.20,
    'rare_kanji': 0.15,
    'inverted_syntax': 0.15,
    'pivot_word': 0.30,
    'allusion': 0.25,
    'compression': 0.20,
}


class DifficultyFactor(BaseModel):
    factor: str
    weight: float = Field(..., ge=0.0, le=1.0)
    note: Optional[str] = None  # LLM explanation for the weight


def compute_difficulty_score(factors: list[DifficultyFactor]) -> float:
    """
    Deterministic difficulty score computation.
    Formula: 1 - product(1 - weight_i) for all factors, capped at 1.0
    This gives diminishing returns for multiple factors.
    """
    if not factors:
        return 0.0
    complement_product = 1.0
    for f in factors:
        complement_product *= (1.0 - f.weight)
    return min(1.0, 1.0 - complement_product)


# -----------------------------------------------------------------------------
# Main annotation schema
# -----------------------------------------------------------------------------

class PoemAnnotation(BaseModel):
    """Complete annotation for a poem."""
    # Identity
    poem_id: str
    text: str
    text_hash: str  # SHA256 prefix for integrity checking
    source: str
    author: Optional[str] = None
    collection: Optional[str] = None

    # Tokenization (Fugashi — source of truth)
    fugashi_tokens: list[FugashiToken]

    # LLM-provided readings (aligned to fugashi_tokens)
    token_readings: list[TokenReading]
    reading_hiragana: str
    reading_romaji: str

    # Grammar analysis
    grammar_points: list[GrammarPoint]

    # Vocabulary for learners
    vocabulary: list[VocabularyAnnotation] = []

    # Difficulty
    difficulty_factors: list[DifficultyFactor]
    difficulty_score_computed: float = Field(default=0.0)  # deterministic, from factors

    # Context
    semantic_notes: Optional[str] = None

    @field_validator('text_hash')
    @classmethod
    def hash_format_valid(cls, v):
        if len(v) < 16 or not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError('text_hash must be at least 16 hex characters')
        return v.lower()

    def model_post_init(self, __context):
        """Compute difficulty score after initialization."""
        object.__setattr__(
            self,
            'difficulty_score_computed',
            compute_difficulty_score(self.difficulty_factors)
        )

    @classmethod
    def compute_text_hash(cls, text: str) -> str:
        """Compute SHA256 hash prefix for a text string."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

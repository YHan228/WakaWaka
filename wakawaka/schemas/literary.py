"""
Literary Analysis Schemas - Pydantic models for poem literary analysis.

These schemas are designed for objective, professional literary analysis,
not forced appreciation. Analysis depth adapts to each poem.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PoeticDevice(BaseModel):
    """A poetic device or technique identified in the poem."""

    name: str = Field(
        description="Device name with Japanese term, e.g., '掛詞 kakekotoba'"
    )
    location: Optional[str] = Field(
        default=None,
        description="The word/phrase where this device appears"
    )
    explanation: str = Field(
        description="How this device functions in this poem"
    )
    effect: Optional[str] = Field(
        default=None,
        description="What artistic effect it achieves"
    )


class Allusion(BaseModel):
    """A literary or cultural allusion in the poem."""

    reference: str = Field(
        description="What is being referenced"
    )
    connection: str = Field(
        description="How this allusion functions in the poem"
    )


class ImageryNote(BaseModel):
    """Analysis of a key image or symbol."""

    image: str = Field(
        description="The image or symbol"
    )
    significance: str = Field(
        description="Its function in the poem"
    )


class PoemLiteraryAnalysis(BaseModel):
    """
    Literary analysis of a classical Japanese poem.

    Provides objective, professional analysis focusing on
    techniques and their function. Not every poem warrants
    deep analysis—simple poems get simple analysis.
    """

    poem_id: str = Field(description="Unique poem identifier")
    text: str = Field(description="The poem text")

    # === REQUIRED - Core Analysis ===
    interpretation: str = Field(
        description="2-4 sentences: What the poem depicts or expresses"
    )
    emotional_tone: str = Field(
        description="1-2 sentences: The mood or emotional register"
    )
    literary_techniques: str = Field(
        description="2-4 sentences: Key techniques used and how they function"
    )

    # === OPTIONAL - Detailed Analysis (include if notable) ===
    imagery_analysis: Optional[list[ImageryNote]] = Field(
        default=None,
        description="Key images and their function"
    )
    structure_and_flow: Optional[str] = Field(
        default=None,
        description="How the poem is constructed"
    )
    sound_and_rhythm: Optional[str] = Field(
        default=None,
        description="Notable sonic qualities"
    )

    # === Poetic Devices (can be empty) ===
    poetic_devices: list[PoeticDevice] = Field(
        default_factory=list,
        description="Classical Japanese poetic techniques identified"
    )

    # === OPTIONAL - Cultural Context ===
    seasonal_context: Optional[str] = Field(
        default=None,
        description="Seasonal references and their associations"
    )
    historical_background: Optional[str] = Field(
        default=None,
        description="Relevant context about composition"
    )
    allusions: list[Allusion] = Field(
        default_factory=list,
        description="Literary or cultural references"
    )
    cultural_notes: Optional[str] = Field(
        default=None,
        description="Context helpful for Chinese readers"
    )

    # === OPTIONAL - Comparative ===
    chinese_poetry_parallel: Optional[str] = Field(
        default=None,
        description="Similar themes or techniques in Chinese poetry"
    )

    # === OPTIONAL - Critical Assessment ===
    critical_notes: Optional[str] = Field(
        default=None,
        description="Honest assessment—strengths, limitations, or conventions"
    )


class LiteraryAnalysisBatch(BaseModel):
    """Container for batch of literary analyses with metadata."""

    analyses: list[PoemLiteraryAnalysis]
    model_used: str
    prompt_version: str
    generated_at: str


# -----------------------------------------------------------------------------
# Literary Index Schemas (for curriculum integration)
# -----------------------------------------------------------------------------


class LiteraryIndexEntry(BaseModel):
    """
    Index entry for a poetic device used in curriculum planning.

    Tracks device frequency, co-occurrence patterns, and example poems
    to inform literary-aware lesson design.
    """

    device_name: str = Field(description="Normalized device name in Japanese, e.g., '掛詞'")
    frequency: int = Field(description="Number of poems containing this device")
    co_occurring_devices: dict[str, int] = Field(
        default_factory=dict,
        description="Other devices that appear in the same poems: {device_name: count}"
    )
    co_occurring_grammar: dict[str, int] = Field(
        default_factory=dict,
        description="Grammar points that co-occur with this device: {canonical_id: count}"
    )
    example_poem_ids: list[str] = Field(
        default_factory=list,
        description="Up to 5 example poem IDs for reference"
    )


class LiteraryIndex(BaseModel):
    """
    Index of all poetic devices in the corpus.

    Used by curriculum extraction to:
    - Select poems with specific literary features
    - Plan literary progression in lessons
    - Identify grammar-literary correlations
    """

    entries: dict[str, LiteraryIndexEntry] = Field(
        description="Device name -> index entry mapping"
    )
    generated_at: str
    corpus_size: int


class PoemLiteraryDifficulty(BaseModel):
    """Literary difficulty assessment for a single poem."""

    poem_id: str
    literary_difficulty: float = Field(
        ge=0.0, le=1.0,
        description="Overall literary complexity score (0=simple, 1=complex)"
    )
    device_count: int = Field(description="Number of poetic devices identified")
    has_complex_device: bool = Field(
        description="True if poem contains 掛詞, 本歌取り, or 序詞"
    )
    interpretation_complexity: float = Field(
        ge=0.0, le=1.0,
        description="Complexity of meaning/interpretation"
    )

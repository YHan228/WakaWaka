"""
Schema validation tests for WakaDecoder.

Tests all Pydantic models to ensure they validate correctly.
"""

import pytest
from datetime import datetime

from wakawaka.schemas import (
    # Annotation
    FugashiToken,
    TokenReading,
    GrammarPoint,
    VocabularyAnnotation,
    DifficultyFactor,
    PoemAnnotation,
    compute_difficulty_score,
    validate_span,
    # Curriculum
    SenseEntry,
    GrammarIndexEntry,
    GrammarIndex,
    PrerequisiteEdge,
    PrerequisiteGraph,
    LessonNode,
    Unit,
    LessonGraph,
    # Lesson
    GrammarExplanation,
    PoemDisplay,
    VocabularyItem,
    IntroductionStep,
    PoemPresentationStep,
    GrammarSpotlightStep,
    ComprehensionCheckStep,
    SummaryStep,
    ReferenceCard,
    LessonContent,
    # Progress
    LessonStatus,
    LessonProgress,
    StudentProgress,
)


class TestSpanValidation:
    """Test span validation helper."""

    def test_valid_span(self):
        assert validate_span([0, 3]) == [0, 3]
        assert validate_span([5, 10]) == [5, 10]

    def test_invalid_span_negative_start(self):
        with pytest.raises(ValueError):
            validate_span([-1, 3])

    def test_invalid_span_end_not_greater(self):
        with pytest.raises(ValueError):
            validate_span([3, 3])
        with pytest.raises(ValueError):
            validate_span([5, 3])

    def test_invalid_span_wrong_length(self):
        with pytest.raises(ValueError):
            validate_span([1])
        with pytest.raises(ValueError):
            validate_span([1, 2, 3])


class TestAnnotationSchemas:
    """Test annotation-related schemas."""

    def test_fugashi_token_valid(self):
        token = FugashiToken(
            surface="古",
            pos="名詞",
            pos_detail="名詞,普通名詞,一般",
            lemma="古",
            span=[0, 1]
        )
        assert token.surface == "古"
        assert token.span == [0, 1]

    def test_token_reading_valid(self):
        reading = TokenReading(token_index=0, reading_kana="ふる")
        assert reading.token_index == 0
        assert reading.reading_kana == "ふる"

    def test_token_reading_invalid_index(self):
        with pytest.raises(ValueError):
            TokenReading(token_index=-1, reading_kana="ふる")

    def test_grammar_point_valid(self):
        gp = GrammarPoint(
            canonical_id="particle_ya",
            sense_id="exclamatory",
            surface="や",
            category="kireji",
            description="Exclamatory kireji",
            span=[2, 3]
        )
        assert gp.canonical_id == "particle_ya"
        assert gp.full_id == "particle_ya_exclamatory"

    def test_grammar_point_no_sense(self):
        gp = GrammarPoint(
            canonical_id="particle_ya",
            surface="や",
            category="kireji",
            description="Kireji",
            span=[2, 3]
        )
        assert gp.full_id == "particle_ya"

    def test_grammar_point_invalid_category(self):
        with pytest.raises(ValueError):
            GrammarPoint(
                canonical_id="particle_ya",
                surface="や",
                category="invalid_category",
                description="Test",
                span=[2, 3]
            )

    def test_vocabulary_annotation_valid(self):
        vocab = VocabularyAnnotation(
            word="飛び込む",
            reading="とびこむ",
            span=[4, 8],
            meaning="to jump into",
            chinese_cognate_note="飛+込 partially guessable"
        )
        assert vocab.word == "飛び込む"

    def test_difficulty_factor_valid(self):
        factor = DifficultyFactor(
            factor="classical_auxiliary",
            weight=0.25,
            note="けり ending"
        )
        assert factor.weight == 0.25

    def test_difficulty_factor_weight_bounds(self):
        with pytest.raises(ValueError):
            DifficultyFactor(factor="test", weight=1.5)
        with pytest.raises(ValueError):
            DifficultyFactor(factor="test", weight=-0.1)

    def test_compute_difficulty_score_empty(self):
        assert compute_difficulty_score([]) == 0.0

    def test_compute_difficulty_score_single(self):
        factors = [DifficultyFactor(factor="test", weight=0.25)]
        assert compute_difficulty_score(factors) == 0.25

    def test_compute_difficulty_score_multiple(self):
        factors = [
            DifficultyFactor(factor="a", weight=0.25),
            DifficultyFactor(factor="b", weight=0.20),
        ]
        # 1 - (1-0.25)*(1-0.20) = 1 - 0.75*0.80 = 1 - 0.60 = 0.40
        assert abs(compute_difficulty_score(factors) - 0.40) < 0.001

    def test_poem_annotation_valid(self):
        annotation = PoemAnnotation(
            poem_id="test_001",
            text="古池や",
            text_hash="a" * 16,
            source="test",
            fugashi_tokens=[
                FugashiToken(surface="古", pos="名詞", pos_detail="名詞,普通名詞", lemma="古", span=[0, 1]),
                FugashiToken(surface="池", pos="名詞", pos_detail="名詞,普通名詞", lemma="池", span=[1, 2]),
                FugashiToken(surface="や", pos="助詞", pos_detail="助詞,終助詞", lemma="や", span=[2, 3]),
            ],
            token_readings=[
                TokenReading(token_index=0, reading_kana="ふる"),
                TokenReading(token_index=1, reading_kana="いけ"),
                TokenReading(token_index=2, reading_kana="や"),
            ],
            reading_hiragana="ふるいけや",
            reading_romaji="furuikeya",
            grammar_points=[
                GrammarPoint(
                    canonical_id="kireji_ya",
                    surface="や",
                    category="kireji",
                    description="Exclamatory kireji",
                    span=[2, 3]
                )
            ],
            difficulty_factors=[
                DifficultyFactor(factor="kireji_ya", weight=0.15)
            ]
        )
        assert annotation.poem_id == "test_001"
        assert abs(annotation.difficulty_score_computed - 0.15) < 0.001

    def test_poem_annotation_text_hash_validation(self):
        with pytest.raises(ValueError):
            PoemAnnotation(
                poem_id="test",
                text="test",
                text_hash="short",  # too short
                source="test",
                fugashi_tokens=[],
                token_readings=[],
                reading_hiragana="",
                reading_romaji="",
                grammar_points=[],
                difficulty_factors=[]
            )

    def test_compute_text_hash(self):
        hash1 = PoemAnnotation.compute_text_hash("古池や")
        assert len(hash1) == 16
        assert all(c in '0123456789abcdef' for c in hash1)


class TestCurriculumSchemas:
    """Test curriculum-related schemas."""

    def test_sense_entry_valid(self):
        entry = SenseEntry(
            sense_id="location",
            surfaces=["に"],
            frequency=50,
            avg_difficulty=0.3,
            example_poem_ids=["poem1", "poem2"]
        )
        assert entry.sense_id == "location"

    def test_grammar_index_entry_valid(self):
        entry = GrammarIndexEntry(
            canonical_id="particle_ni",
            category="particle",
            surfaces=["に"],
            frequency=100,
            avg_difficulty=0.25,
            senses=[
                SenseEntry(sense_id="location", surfaces=["に"], frequency=50, avg_difficulty=0.2),
                SenseEntry(sense_id="time", surfaces=["に"], frequency=30, avg_difficulty=0.3),
            ],
            co_occurrences={"particle_wa": 80, "particle_ga": 60}
        )
        assert entry.canonical_id == "particle_ni"
        assert len(entry.senses) == 2

    def test_prerequisite_edge_valid(self):
        edge = PrerequisiteEdge(
            from_id="particle_wa",
            to_id="auxiliary_keri",
            co_ratio=0.75,
            difficulty_gap=0.12,
            support_count=8
        )
        assert edge.from_id == "particle_wa"

    def test_lesson_node_valid(self):
        node = LessonNode(
            id="lesson_particle_wa",
            canonical_grammar_point="particle_wa",
            senses_covered=["topic", "contrast"],
            prerequisites=[],
            difficulty_tier=1,
            poem_ids=["poem1", "poem2"]
        )
        assert node.id == "lesson_particle_wa"

    def test_lesson_node_difficulty_tier_bounds(self):
        with pytest.raises(ValueError):
            LessonNode(
                id="test",
                canonical_grammar_point="test",
                prerequisites=[],
                difficulty_tier=0,  # must be >= 1
                poem_ids=[]
            )
        with pytest.raises(ValueError):
            LessonNode(
                id="test",
                canonical_grammar_point="test",
                prerequisites=[],
                difficulty_tier=6,  # must be <= 5
                poem_ids=[]
            )

    def test_lesson_graph_valid(self):
        graph = LessonGraph(
            units=[
                Unit(
                    id="unit_particle",
                    lessons=[
                        LessonNode(
                            id="lesson_particle_wa",
                            canonical_grammar_point="particle_wa",
                            prerequisites=[],
                            difficulty_tier=1,
                            poem_ids=["poem1"]
                        )
                    ]
                )
            ],
            prerequisite_graph=PrerequisiteGraph(edges=[]),
            meta={"generated_at": "2024-01-01", "total_lessons": 1}
        )
        assert len(graph.units) == 1


class TestLessonSchemas:
    """Test lesson content schemas."""

    def test_grammar_explanation_valid(self):
        explanation = GrammarExplanation(
            concept="Topic marker",
            formation="Attach to noun",
            variations=["は", "も"],
            common_confusions=["vs が"],
            logic_analogy="Like 'as for' in English"
        )
        assert explanation.concept == "Topic marker"

    def test_poem_display_valid(self):
        display = PoemDisplay(
            text_with_furigana="<ruby>古<rt>ふる</rt></ruby><ruby>池<rt>いけ</rt></ruby>や",
            romaji="furuikeya",
            translation="An old pond"
        )
        assert "ruby" in display.text_with_furigana

    def test_teaching_steps_valid(self):
        intro = IntroductionStep(content="Welcome to this lesson")
        assert intro.type == "introduction"

        poem = PoemPresentationStep(
            poem_id="test_001",
            display=PoemDisplay(
                text_with_furigana="<ruby>古<rt>ふる</rt></ruby>池や",
                romaji="furuikeya",
                translation="An old pond"
            ),
            vocabulary=[
                VocabularyItem(word="古", reading="ふる", meaning="old")
            ],
            focus_highlight=[2, 3]
        )
        assert poem.type == "poem_presentation"

        spotlight = GrammarSpotlightStep(
            content="Notice the や here",
            evidence="The や at position 2"
        )
        assert spotlight.type == "grammar_spotlight"

        check = ComprehensionCheckStep(
            question="What does や do?",
            answer="It creates a pause",
            hint="Think about kireji"
        )
        assert check.type == "comprehension_check"

        summary = SummaryStep(content="In this lesson we learned...")
        assert summary.type == "summary"

    def test_lesson_content_valid(self):
        lesson = LessonContent(
            lesson_id="lesson_kireji_ya",
            lesson_title="The Kireji や",
            lesson_summary="Learn about the cutting word や",
            grammar_point="kireji_ya",
            grammar_explanation=GrammarExplanation(
                concept="Cutting word for pause/exclamation"
            ),
            teaching_sequence=[
                IntroductionStep(content="Welcome"),
                PoemPresentationStep(
                    poem_id="test_001",
                    display=PoemDisplay(
                        text_with_furigana="古池や",
                        romaji="furuikeya",
                        translation="An old pond"
                    )
                ),
                ComprehensionCheckStep(
                    question="What does や do?",
                    answer="Creates a pause"
                )
            ],
            reference_card=ReferenceCard(
                point="kireji_ya",
                one_liner="Cutting word for pause",
                example="古池や - 'An old pond!'"
            )
        )
        assert lesson.lesson_id == "lesson_kireji_ya"

    def test_lesson_content_requires_min_steps(self):
        with pytest.raises(ValueError):
            LessonContent(
                lesson_id="test",
                lesson_title="Test",
                lesson_summary="Test",
                grammar_point="test",
                grammar_explanation=GrammarExplanation(concept="Test"),
                teaching_sequence=[
                    IntroductionStep(content="Only one step")
                ],  # needs at least 3
                reference_card=ReferenceCard(point="test", one_liner="test", example="test")
            )


class TestProgressSchemas:
    """Test progress tracking schemas."""

    def test_lesson_status_values(self):
        assert LessonStatus.NOT_STARTED.value == "not_started"
        assert LessonStatus.IN_PROGRESS.value == "in_progress"
        assert LessonStatus.COMPLETED.value == "completed"

    def test_lesson_progress_valid(self):
        progress = LessonProgress(
            lesson_id="lesson_001",
            status=LessonStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 10, 0),
            completed_at=datetime(2024, 1, 1, 10, 30),
            quiz_score=0.85
        )
        assert progress.lesson_id == "lesson_001"
        assert progress.status == LessonStatus.COMPLETED

    def test_lesson_progress_defaults(self):
        progress = LessonProgress(lesson_id="lesson_001")
        assert progress.status == LessonStatus.NOT_STARTED
        assert progress.started_at is None

    def test_student_progress_valid(self):
        student = StudentProgress(
            student_id="user_001",
            lessons={
                "lesson_001": LessonProgress(
                    lesson_id="lesson_001",
                    status=LessonStatus.COMPLETED
                ),
                "lesson_002": LessonProgress(
                    lesson_id="lesson_002",
                    status=LessonStatus.IN_PROGRESS
                )
            },
            current_lesson_id="lesson_002",
            total_time_minutes=45
        )
        assert student.student_id == "user_001"
        assert len(student.lessons) == 2

    def test_student_progress_defaults(self):
        student = StudentProgress(lessons={})
        assert student.student_id == "default"
        assert student.total_time_minutes == 0


class TestSchemaImports:
    """Test that all schemas can be imported from the main module."""

    def test_import_from_wakawaka_schemas(self):
        from wakawaka.schemas import (
            PoemAnnotation,
            LessonGraph,
            LessonContent,
            StudentProgress,
        )
        assert PoemAnnotation is not None
        assert LessonGraph is not None
        assert LessonContent is not None
        assert StudentProgress is not None

"""
WakaDecoder - Classical Japanese Poetry Learning Platform

Streamlit application for learning classical Japanese through poetry.
Designed for Chinese speakers leveraging kanji knowledge.

Usage:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path

from wakadecoder.classroom import (
    ClassroomLoader,
    ProgressTracker,
    Navigator,
    LessonAvailability,
)
from wakadecoder.viewer import (
    render_lesson,
    get_vocab_css,
    extract_quiz_questions,
    render_quiz_question,
    calculate_quiz_score,
    render_quiz_score,
    render_reference_card_from_lesson,
    render_grammar_index,
    filter_grammar_points,
    get_reference_css,
)
from wakadecoder.schemas import LessonStatus


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DB_PATH = Path("data/classroom.db")

st.set_page_config(
    page_title="WakaDecoder",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

def init_session_state():
    """Initialize session state variables."""
    if "loader" not in st.session_state:
        if DEFAULT_DB_PATH.exists():
            st.session_state.loader = ClassroomLoader(DEFAULT_DB_PATH)
        else:
            st.session_state.loader = None

    if "progress" not in st.session_state:
        st.session_state.progress = ProgressTracker()

    if "navigator" not in st.session_state and st.session_state.loader:
        st.session_state.navigator = Navigator(
            st.session_state.loader,
            st.session_state.progress,
        )

    if "current_lesson_id" not in st.session_state:
        if st.session_state.loader:
            st.session_state.current_lesson_id = st.session_state.navigator.get_recommended_lesson_id()
        else:
            st.session_state.current_lesson_id = None

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "lesson"  # lesson, reference, poems

    if "quiz_revealed" not in st.session_state:
        st.session_state.quiz_revealed = set()


# -----------------------------------------------------------------------------
# Sidebar: Curriculum Tree
# -----------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with curriculum tree and progress."""
    st.sidebar.title("📜 WakaDecoder")

    if not st.session_state.loader:
        st.sidebar.error("Database not found. Please run the build pipeline first.")
        return

    nav = st.session_state.navigator
    progress = st.session_state.progress

    # Progress summary
    stats = nav.get_progress_summary()
    st.sidebar.markdown(f"""
    **Progress:** {stats['completed']}/{stats['total_lessons']} lessons ({stats['completion_percent']}%)
    """)
    st.sidebar.progress(stats['completion_percent'] / 100)

    st.sidebar.divider()

    # View mode selector
    st.sidebar.subheader("View Mode")
    view_mode = st.sidebar.radio(
        "Select view",
        ["Lessons", "Reference", "Poems"],
        index=["lesson", "reference", "poems"].index(st.session_state.view_mode),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.view_mode = view_mode.lower()

    if st.session_state.view_mode == "lesson":
        render_curriculum_tree()


def render_curriculum_tree():
    """Render the curriculum tree with lesson navigation."""
    nav = st.session_state.navigator
    tree = nav.get_navigation_tree()

    st.sidebar.divider()
    st.sidebar.subheader("Curriculum")

    for nav_unit in tree:
        unit = nav_unit.unit
        # Unit header with completion status
        unit_progress = f"({nav_unit.completed_count}/{nav_unit.total_count})"
        with st.sidebar.expander(f"**{unit.title}** {unit_progress}", expanded=is_unit_expanded(unit.id)):
            for nav_lesson in nav_unit.lessons:
                lesson = nav_lesson.summary
                indicator = nav.get_status_indicator(lesson.id)

                # Style based on availability
                if nav_lesson.availability == LessonAvailability.LOCKED:
                    style = "color: #999;"
                    disabled = True
                elif nav_lesson.availability == LessonAvailability.COMPLETED:
                    style = "color: #388E3C;"
                    disabled = False
                elif nav_lesson.is_current:
                    style = "color: #1976D2; font-weight: bold;"
                    disabled = False
                else:
                    style = ""
                    disabled = False

                # Lesson button
                col1, col2 = st.columns([1, 9])
                with col1:
                    st.markdown(f"<span style='{style}'>{indicator}</span>", unsafe_allow_html=True)
                with col2:
                    if st.button(
                        lesson.title[:30] + "..." if len(lesson.title) > 30 else lesson.title,
                        key=f"lesson_{lesson.id}",
                        disabled=disabled,
                        use_container_width=True,
                    ):
                        select_lesson(lesson.id)


def is_unit_expanded(unit_id: str) -> bool:
    """Check if a unit should be expanded (contains current lesson)."""
    current_id = st.session_state.current_lesson_id
    if not current_id:
        return unit_id == "unit_01"

    lesson = st.session_state.loader.get_lesson_summary(current_id)
    return lesson and lesson.unit_id == unit_id


def select_lesson(lesson_id: str):
    """Select a lesson and update state."""
    st.session_state.current_lesson_id = lesson_id
    st.session_state.quiz_revealed = set()
    st.session_state.navigator.start_lesson(lesson_id)
    st.rerun()


# -----------------------------------------------------------------------------
# Main Content: Lesson View
# -----------------------------------------------------------------------------

def render_lesson_view():
    """Render the main lesson content."""
    if not st.session_state.loader:
        st.error("Database not found. Please run the build pipeline first.")
        st.code("""
# Build the database:
python scripts/01_ingest_corpus.py --source all
python scripts/02_annotate_corpus.py --input-dir data/raw --output data/annotated/poems.parquet
python scripts/03_extract_curriculum.py --input data/annotated/poems.parquet --output-dir data/curriculum
python scripts/04_generate_lessons.py --curriculum data/curriculum --all
python scripts/05_compile_classroom.py --lessons data/lessons --curriculum data/curriculum --poems data/annotated/poems.parquet --output data/classroom.db
        """)
        return

    lesson_id = st.session_state.current_lesson_id
    if not lesson_id:
        st.info("Select a lesson from the sidebar to begin.")
        return

    nav = st.session_state.navigator
    loader = st.session_state.loader
    progress = st.session_state.progress

    # Load lesson content
    lesson = loader.get_lesson_content(lesson_id)
    if not lesson:
        st.error(f"Lesson not found: {lesson_id}")
        return

    # Navigation bar
    render_navigation_bar(lesson_id)

    # Lesson content
    st.markdown(get_vocab_css(), unsafe_allow_html=True)
    st.markdown(render_lesson(lesson), unsafe_allow_html=True)

    # Quiz section
    render_quiz_section(lesson)

    # Reference card
    st.divider()
    st.subheader("Reference Card")
    st.markdown(get_reference_css(), unsafe_allow_html=True)
    st.markdown(render_reference_card_from_lesson(lesson), unsafe_allow_html=True)

    # Completion button
    render_completion_section(lesson_id)


def render_navigation_bar(lesson_id: str):
    """Render navigation bar with prev/next buttons."""
    nav = st.session_state.navigator
    pos, total = nav.get_lesson_position(lesson_id)

    prev_id = nav.get_previous_lesson_id(lesson_id)
    next_id = nav.get_next_lesson_id(lesson_id)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if prev_id and nav.is_lesson_available(prev_id):
            if st.button("← Previous", use_container_width=True):
                select_lesson(prev_id)

    with col2:
        st.markdown(f"<center>Lesson {pos} of {total}</center>", unsafe_allow_html=True)

    with col3:
        if next_id and nav.is_lesson_available(next_id):
            if st.button("Next →", use_container_width=True):
                select_lesson(next_id)

    st.divider()


def render_quiz_section(lesson):
    """Render comprehension check section."""
    quizzes = extract_quiz_questions(lesson)
    if not quizzes:
        return

    st.divider()
    st.subheader("Comprehension Check")

    for quiz in quizzes:
        with st.container():
            st.markdown(f"**Question {quiz.index + 1}:** {quiz.question}")

            if quiz.hint:
                with st.expander("Show hint"):
                    st.info(quiz.hint)

            key = f"quiz_{lesson.lesson_id}_{quiz.index}"
            revealed = quiz.index in st.session_state.quiz_revealed

            if revealed:
                st.success(f"**Answer:** {quiz.answer}")
            else:
                if st.button("Reveal Answer", key=key):
                    st.session_state.quiz_revealed.add(quiz.index)
                    st.rerun()


def render_completion_section(lesson_id: str):
    """Render lesson completion section."""
    progress = st.session_state.progress
    nav = st.session_state.navigator

    lesson_progress = progress.get_lesson_progress(lesson_id)

    st.divider()

    if lesson_progress.status == LessonStatus.COMPLETED:
        st.success("Lesson completed!")
        if st.button("Mark as incomplete"):
            progress.reset_lesson(lesson_id)
            st.rerun()
    else:
        if st.button("Mark lesson as complete", type="primary", use_container_width=True):
            next_id = nav.complete_lesson(lesson_id)
            st.success("Lesson completed!")
            if next_id:
                st.session_state.current_lesson_id = next_id
            st.rerun()


# -----------------------------------------------------------------------------
# Reference View
# -----------------------------------------------------------------------------

def render_reference_view():
    """Render grammar reference mode."""
    if not st.session_state.loader:
        st.error("Database not found.")
        return

    st.title("Grammar Reference")

    loader = st.session_state.loader
    progress = st.session_state.progress

    # Search box
    search = st.text_input("Search grammar points", placeholder="e.g., particle, verb, auxiliary")

    # Get grammar points
    grammar_points = loader.get_all_grammar_points()

    if search:
        grammar_points = filter_grammar_points(grammar_points, search)

    st.markdown(f"**{len(grammar_points)} grammar points**")

    # Tabs for different views
    tab1, tab2 = st.tabs(["Grammar Index", "Learned Reference Cards"])

    with tab1:
        st.markdown(get_reference_css(), unsafe_allow_html=True)
        st.markdown(render_grammar_index(grammar_points), unsafe_allow_html=True)

    with tab2:
        # Show reference cards for completed lessons
        completed_ids = progress.get_completed_lesson_ids()
        if not completed_ids:
            st.info("Complete lessons to build your reference card collection.")
        else:
            st.markdown(get_reference_css(), unsafe_allow_html=True)
            for lesson_id in completed_ids:
                lesson = loader.get_lesson_content(lesson_id)
                if lesson:
                    st.markdown(render_reference_card_from_lesson(lesson), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Poems View
# -----------------------------------------------------------------------------

def render_poems_view():
    """Render poem anthology view."""
    if not st.session_state.loader:
        st.error("Database not found.")
        return

    st.title("Poem Anthology")

    loader = st.session_state.loader

    # Get metadata
    metadata = loader.get_all_metadata()
    total_poems = loader.get_poem_count()

    st.markdown(f"**{total_poems} poems** from classical Japanese collections")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        source_filter = st.selectbox("Source", ["All", "ogura100", "lapis"])
    with col2:
        difficulty_filter = st.slider("Max Difficulty", 0.0, 1.0, 1.0, 0.1)

    # Load and filter poems
    poems = loader.get_all_poems(limit=100)

    if source_filter != "All":
        poems = [p for p in poems if p.source == source_filter]

    poems = [p for p in poems if p.difficulty_score <= difficulty_filter]

    st.markdown(f"Showing {len(poems)} poems")

    # Display poems
    for poem in poems[:20]:  # Limit display
        with st.expander(f"{poem.text[:30]}... ({poem.source})"):
            st.markdown(f"**Text:** {poem.text}")
            if poem.reading_hiragana:
                st.markdown(f"**Reading:** {poem.reading_hiragana}")
            if poem.reading_romaji:
                st.markdown(f"**Romaji:** {poem.reading_romaji}")
            if poem.author:
                st.markdown(f"**Author:** {poem.author}")
            if poem.collection:
                st.markdown(f"**Collection:** {poem.collection}")
            st.markdown(f"**Difficulty:** {poem.difficulty_score:.2f}")


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Main content based on view mode
    if st.session_state.view_mode == "lesson":
        render_lesson_view()
    elif st.session_state.view_mode == "reference":
        render_reference_view()
    elif st.session_state.view_mode == "poems":
        render_poems_view()


if __name__ == "__main__":
    main()

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


def inject_global_css():
    """Inject global CSS for the Ink & Paper theme."""
    st.markdown("""
    <style>
    /* ============================================
       GLOBAL APP STYLING - Ink & Paper Theme
       ============================================ */

    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* -- Root Variables -- */
    :root {
        --washi-cream: #FAF8F5;
        --washi-warm: #F5F2ED;
        --sumi-ink: #2D2D2D;
        --sumi-light: #4A4A4A;
        --vermillion: #C53D43;
        --indigo: #4A5568;
        --pine-green: #5B8A72;
        --plum: #8B687F;
        --gold-accent: #D4A84B;
        --soft-blue: #6B8CAE;
    }

    /* -- Main App Background -- */
    .stApp {
        background: linear-gradient(180deg, var(--washi-cream) 0%, #F8F6F3 100%);
    }

    /* -- Sidebar Styling -- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2D2D2D 0%, #3D3D3D 100%);
    }

    [data-testid="stSidebar"] * {
        color: #E8E4DE !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #FAF8F5 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1);
    }

    [data-testid="stSidebar"] .stProgress > div > div {
        background-color: var(--vermillion);
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] button:hover {
        background: rgba(255,255,255,0.1) !important;
        border-color: var(--vermillion) !important;
    }

    /* Sidebar expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
    }

    /* -- Main Content Typography -- */
    .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stMarkdown h1 {
        font-family: 'Noto Serif JP', serif;
        color: var(--sumi-ink);
        font-weight: 700;
        border-bottom: 2px solid var(--vermillion);
        padding-bottom: 0.5em;
        margin-bottom: 1em;
    }

    .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Noto Serif JP', serif;
        color: var(--sumi-ink);
    }

    /* -- Navigation Bar -- */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1em 0;
        margin-bottom: 1em;
        border-bottom: 1px solid #E8E4DE;
    }

    .nav-position {
        font-size: 0.9em;
        color: var(--sumi-light);
    }

    /* -- Quiz Section -- */
    .quiz-section {
        background: linear-gradient(135deg, #F8F6F3 0%, #F5F2ED 100%);
        border: 1px solid #E8E4DE;
        border-radius: 12px;
        padding: 1.5em 2em;
        margin: 1.5em 0;
    }

    .quiz-header {
        display: flex;
        align-items: center;
        gap: 0.5em;
        margin-bottom: 1em;
        color: var(--indigo);
    }

    .quiz-question {
        font-size: 1.05em;
        line-height: 1.7;
        color: var(--sumi-ink);
        margin-bottom: 1em;
    }

    .quiz-hint {
        background: rgba(212, 168, 75, 0.15);
        border-left: 3px solid var(--gold-accent);
        padding: 0.8em 1em;
        border-radius: 0 8px 8px 0;
        margin: 1em 0;
        font-size: 0.95em;
        color: var(--sumi-light);
    }

    .quiz-answer-box {
        background: rgba(91, 138, 114, 0.1);
        border: 1px solid var(--pine-green);
        border-radius: 8px;
        padding: 1em 1.2em;
        margin-top: 1em;
    }

    .quiz-answer-label {
        font-weight: 600;
        color: var(--pine-green);
        margin-bottom: 0.5em;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .quiz-answer-content {
        color: var(--sumi-ink);
        line-height: 1.6;
    }

    .self-assess {
        display: flex;
        gap: 0.5em;
        margin-top: 1em;
        padding-top: 1em;
        border-top: 1px dashed #E8E4DE;
    }

    /* -- Completion Section -- */
    .completion-section {
        background: linear-gradient(135deg, var(--washi-cream) 0%, #F8F6F3 100%);
        border: 2px solid #E8E4DE;
        border-radius: 12px;
        padding: 1.5em;
        margin: 2em 0;
        text-align: center;
    }

    /* -- Buttons -- */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--vermillion) !important;
        border-color: var(--vermillion) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #A83238 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(197, 61, 67, 0.3);
    }

    /* -- Text Input -- */
    .stTextArea textarea {
        font-family: 'Inter', sans-serif;
        border: 2px solid #E8E4DE;
        border-radius: 8px;
        padding: 0.8em;
        transition: border-color 0.2s;
    }

    .stTextArea textarea:focus {
        border-color: var(--soft-blue);
        box-shadow: 0 0 0 3px rgba(107, 140, 174, 0.1);
    }

    /* -- Tabs -- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5em;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    /* -- Expander -- */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    /* -- Poem Display in Anthology -- */
    .poem-anthology-item {
        background: var(--washi-cream);
        border: 1px solid #E8E4DE;
        border-radius: 8px;
        padding: 1.2em;
        margin: 0.5em 0;
    }

    .poem-anthology-text {
        font-family: 'Noto Serif JP', serif;
        font-size: 1.1em;
        color: var(--sumi-ink);
        margin-bottom: 0.5em;
    }

    .poem-anthology-meta {
        font-size: 0.85em;
        color: var(--sumi-light);
    }

    </style>
    """, unsafe_allow_html=True)


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
        st.session_state.view_mode = "lesson"

    if "quiz_revealed" not in st.session_state:
        st.session_state.quiz_revealed = set()

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with curriculum tree and progress."""
    with st.sidebar:
        st.markdown("# 📜 WakaDecoder")
        st.caption("Classical Japanese Poetry")

        if not st.session_state.loader:
            st.error("Database not found")
            return

        nav = st.session_state.navigator

        # Progress summary
        stats = nav.get_progress_summary()
        st.markdown(f"**Progress:** {stats['completed']}/{stats['total_lessons']}")
        st.progress(stats['completion_percent'] / 100)

        st.divider()

        # View mode selector
        mode_map = {"lesson": 0, "reference": 1, "poems": 2}
        view_mode = st.radio(
            "View Mode",
            ["📖 Lessons", "📚 Reference", "🎋 Poems"],
            index=mode_map.get(st.session_state.view_mode, 0),
            horizontal=False,
        )
        reverse_map = {"📖 Lessons": "lesson", "📚 Reference": "reference", "🎋 Poems": "poems"}
        st.session_state.view_mode = reverse_map.get(view_mode, "lesson")

        if st.session_state.view_mode == "lesson":
            st.divider()
            render_curriculum_tree()


def render_curriculum_tree():
    """Render the curriculum tree with lesson navigation."""
    nav = st.session_state.navigator
    tree = nav.get_navigation_tree()

    for nav_unit in tree:
        unit = nav_unit.unit
        progress_pct = int(nav_unit.completed_count / nav_unit.total_count * 100) if nav_unit.total_count > 0 else 0

        # Unit header
        with st.expander(f"**{unit.title}** ({progress_pct}%)", expanded=is_unit_expanded(unit.id)):
            for nav_lesson in nav_unit.lessons:
                lesson = nav_lesson.summary
                indicator = nav.get_status_indicator(lesson.id)

                # Determine button state
                disabled = nav_lesson.availability == LessonAvailability.LOCKED
                is_current = nav_lesson.is_current

                # Format title
                title = lesson.title[:35] + "…" if len(lesson.title) > 35 else lesson.title

                # Single button with indicator
                btn_label = f"{indicator} {title}"
                if st.button(
                    btn_label,
                    key=f"lesson_{lesson.id}",
                    disabled=disabled,
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    select_lesson(lesson.id)


def is_unit_expanded(unit_id: str) -> bool:
    """Check if a unit should be expanded."""
    current_id = st.session_state.current_lesson_id
    if not current_id:
        return unit_id == "unit_01"
    lesson = st.session_state.loader.get_lesson_summary(current_id)
    return lesson and lesson.unit_id == unit_id


def select_lesson(lesson_id: str):
    """Select a lesson and update state."""
    st.session_state.current_lesson_id = lesson_id
    st.session_state.quiz_revealed = set()
    st.session_state.quiz_answers = {}
    st.session_state.navigator.start_lesson(lesson_id)
    st.rerun()


# -----------------------------------------------------------------------------
# Main Content: Lesson View
# -----------------------------------------------------------------------------

def render_lesson_view():
    """Render the main lesson content."""
    if not st.session_state.loader:
        st.error("Database not found. Please run the build pipeline first.")
        with st.expander("Build Instructions"):
            st.code("""
python scripts/01_ingest_corpus.py --source all
python scripts/02_annotate_corpus.py --input-dir data/raw --output data/annotated/poems.parquet
python scripts/03_extract_curriculum.py --input data/annotated/poems.parquet --output-dir data/curriculum
python scripts/04_generate_lessons.py --curriculum data/curriculum --all
python scripts/05_compile_classroom.py --output data/classroom.db
            """)
        return

    lesson_id = st.session_state.current_lesson_id
    if not lesson_id:
        st.markdown("## Welcome to WakaDecoder")
        st.markdown("Select a lesson from the sidebar to begin your journey into classical Japanese poetry.")
        return

    nav = st.session_state.navigator
    loader = st.session_state.loader

    # Load lesson content
    lesson = loader.get_lesson_content(lesson_id)
    if not lesson:
        st.error(f"Lesson not found: {lesson_id}")
        return

    # Navigation bar
    render_navigation_bar(lesson_id)

    # Inject CSS and render lesson
    st.markdown(get_vocab_css(), unsafe_allow_html=True)
    st.markdown(render_lesson(lesson), unsafe_allow_html=True)

    # Quiz section with text input
    render_quiz_section(lesson)

    # Reference card
    st.divider()
    st.markdown("### 📋 Reference Card")
    st.markdown(get_reference_css(), unsafe_allow_html=True)
    st.markdown(render_reference_card_from_lesson(lesson), unsafe_allow_html=True)

    # Completion
    render_completion_section(lesson_id)


def render_navigation_bar(lesson_id: str):
    """Render navigation bar."""
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
        st.markdown(f"<p style='text-align:center;color:#666;margin-top:0.5em;'>Lesson {pos} of {total}</p>",
                    unsafe_allow_html=True)

    with col3:
        if next_id and nav.is_lesson_available(next_id):
            if st.button("Next →", use_container_width=True):
                select_lesson(next_id)


def render_quiz_section(lesson):
    """Render comprehension check with free text input."""
    quizzes = extract_quiz_questions(lesson)
    if not quizzes:
        return

    st.divider()
    st.markdown("### ✍️ Comprehension Check")

    for quiz in quizzes:
        quiz_key = f"{lesson.lesson_id}_{quiz.index}"

        st.markdown(f"""
        <div class="quiz-section">
            <div class="quiz-header">
                <strong>Question {quiz.index + 1}</strong>
            </div>
            <div class="quiz-question">{quiz.question}</div>
        </div>
        """, unsafe_allow_html=True)

        # Hint expander
        if quiz.hint:
            with st.expander("💡 Show hint"):
                st.markdown(f"""
                <div class="quiz-hint">{quiz.hint}</div>
                """, unsafe_allow_html=True)

        # User answer input
        user_answer = st.text_area(
            "Your answer:",
            key=f"answer_{quiz_key}",
            placeholder="Write your understanding before revealing the answer...",
            height=100,
            label_visibility="collapsed",
        )

        # Store answer in session state
        if user_answer:
            st.session_state.quiz_answers[quiz_key] = user_answer

        # Reveal answer
        revealed = quiz.index in st.session_state.quiz_revealed

        col1, col2 = st.columns([1, 3])
        with col1:
            if not revealed:
                if st.button("Reveal Answer", key=f"reveal_{quiz_key}", type="primary"):
                    st.session_state.quiz_revealed.add(quiz.index)
                    st.rerun()

        if revealed:
            st.markdown(f"""
            <div class="quiz-answer-box">
                <div class="quiz-answer-label">✓ Model Answer</div>
                <div class="quiz-answer-content">{quiz.answer}</div>
            </div>
            """, unsafe_allow_html=True)

            # Self-assessment
            st.markdown("**How did you do?**")
            assess_col1, assess_col2, assess_col3 = st.columns(3)
            with assess_col1:
                st.button("😊 Got it!", key=f"assess_good_{quiz_key}", use_container_width=True)
            with assess_col2:
                st.button("🤔 Partially", key=f"assess_partial_{quiz_key}", use_container_width=True)
            with assess_col3:
                st.button("😅 Need review", key=f"assess_review_{quiz_key}", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)


def render_completion_section(lesson_id: str):
    """Render lesson completion section."""
    progress = st.session_state.progress
    nav = st.session_state.navigator
    lesson_progress = progress.get_lesson_progress(lesson_id)

    st.divider()

    if lesson_progress.status == LessonStatus.COMPLETED:
        st.success("✓ Lesson completed!")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Reset progress"):
                progress.reset_lesson(lesson_id)
                st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✓ Mark Lesson Complete", type="primary", use_container_width=True):
                next_id = nav.complete_lesson(lesson_id)
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

    st.markdown("# 📚 Grammar Reference")

    loader = st.session_state.loader
    progress = st.session_state.progress

    # Search
    search = st.text_input("🔍 Search grammar points", placeholder="e.g., particle, verb, auxiliary")

    grammar_points = loader.get_all_grammar_points()
    if search:
        grammar_points = filter_grammar_points(grammar_points, search)

    st.caption(f"{len(grammar_points)} grammar points")

    # Tabs
    tab1, tab2 = st.tabs(["All Grammar", "My Reference Cards"])

    with tab1:
        st.markdown(get_reference_css(), unsafe_allow_html=True)
        st.markdown(render_grammar_index(grammar_points), unsafe_allow_html=True)

    with tab2:
        completed_ids = progress.get_completed_lesson_ids()
        if not completed_ids:
            st.info("Complete lessons to build your reference card collection.")
        else:
            st.markdown(get_reference_css(), unsafe_allow_html=True)
            for lid in sorted(completed_ids):
                lesson = loader.get_lesson_content(lid)
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

    st.markdown("# 🎋 Poem Anthology")

    loader = st.session_state.loader
    total_poems = loader.get_poem_count()

    st.caption(f"{total_poems} classical Japanese poems")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        source_filter = st.selectbox("Source", ["All", "ogura100", "lapis"])
    with col2:
        difficulty_filter = st.slider("Max Difficulty", 0.0, 1.0, 1.0, 0.1)

    # Load poems
    poems = loader.get_all_poems(limit=100)

    if source_filter != "All":
        poems = [p for p in poems if p.source == source_filter]
    poems = [p for p in poems if p.difficulty_score <= difficulty_filter]

    st.markdown(f"**Showing {len(poems)} poems**")

    # Display
    for poem in poems[:30]:
        with st.expander(f"「{poem.text[:25]}…」"):
            st.markdown(f"""
            <div class="poem-anthology-item">
                <div class="poem-anthology-text">{poem.text}</div>
                <div class="poem-anthology-meta">
                    {f"Author: {poem.author}" if poem.author else ""}
                    {f" | Collection: {poem.collection}" if poem.collection else ""}
                    | Difficulty: {poem.difficulty_score:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if poem.reading_romaji:
                st.caption(f"Romaji: {poem.reading_romaji}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    init_session_state()
    inject_global_css()
    render_sidebar()

    if st.session_state.view_mode == "lesson":
        render_lesson_view()
    elif st.session_state.view_mode == "reference":
        render_reference_view()
    elif st.session_state.view_mode == "poems":
        render_poems_view()


if __name__ == "__main__":
    main()

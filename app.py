"""
WakaWaka - Classical Japanese Poetry Learning Platform

Streamlit application for learning classical Japanese through poetry.
Designed for Chinese speakers leveraging kanji knowledge.

Usage:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path

from wakawaka.classroom import (
    ClassroomLoader,
    LiteraryLoader,
    ProgressTracker,
    Navigator,
    LessonAvailability,
)
from wakawaka.viewer import (
    render_lesson,
    get_vocab_css,
    extract_quiz_questions,
    render_reference_card_from_lesson,
    render_grammar_index,
    filter_grammar_points,
    get_reference_css,
    markdown_to_html,
    get_audio_for_lesson,
    get_audio_path,
    render_teaching_step,
    render_grammar_explanation,
    render_forward_references,
    get_literary_css,
    render_literary_analysis,
)
from wakawaka.schemas import LessonStatus
import json


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DB_PATH = Path("data/classroom.db")
INTRODUCTION_PATH = Path("data/introduction.json")
DATA_DIR = Path("data")
LITERARY_PATH = Path("data/literary/poems_literary.parquet")

st.set_page_config(
    page_title="WakaWaka",
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

    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;500;600;700&family=Noto+Serif+SC:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

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
        font-family: 'Inter', 'Noto Serif SC', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ensure all Chinese text uses proper fonts */
    .stMarkdown p, .stMarkdown li, .stMarkdown blockquote {
        font-family: 'Noto Serif SC', 'Inter', -apple-system, sans-serif;
    }

    .stMarkdown h1 {
        font-family: 'Noto Serif SC', 'Noto Serif JP', serif;
        color: var(--sumi-ink);
        font-weight: 700;
        border-bottom: 2px solid var(--vermillion);
        padding-bottom: 0.5em;
        margin-bottom: 1em;
    }

    .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Noto Serif SC', 'Noto Serif JP', serif;
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
        font-family: 'Noto Serif JP', 'Noto Serif SC', serif;
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

    if "literary_loader" not in st.session_state:
        if LITERARY_PATH.exists():
            st.session_state.literary_loader = LiteraryLoader(LITERARY_PATH)
        else:
            st.session_state.literary_loader = None

    if "progress" not in st.session_state:
        st.session_state.progress = ProgressTracker()

    if "navigator" not in st.session_state and st.session_state.loader:
        st.session_state.navigator = Navigator(
            st.session_state.loader,
            st.session_state.progress,
        )

    if "current_lesson_id" not in st.session_state:
        st.session_state.current_lesson_id = None  # Start with cover page

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "cover"  # Start with cover page

    if "quiz_revealed" not in st.session_state:
        st.session_state.quiz_revealed = set()

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    if "introduction" not in st.session_state:
        if INTRODUCTION_PATH.exists():
            with open(INTRODUCTION_PATH) as f:
                st.session_state.introduction = json.load(f)
        else:
            st.session_state.introduction = None


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with curriculum tree and progress."""
    with st.sidebar:
        st.markdown("# 📜 WakaWaka")
        st.caption("古典和歌学习平台")

        if not st.session_state.loader:
            st.error("Database not found")
            return

        nav = st.session_state.navigator

        # Progress summary
        stats = nav.get_progress_summary()
        st.markdown(f"**学习进度：** {stats['completed']}/{stats['total_lessons']}")
        st.progress(stats['completion_percent'] / 100)

        st.divider()

        # Navigation buttons for cover and intro
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 首页", use_container_width=True,
                        type="primary" if st.session_state.view_mode == "cover" else "secondary"):
                st.session_state.view_mode = "cover"
                st.rerun()
        with col2:
            if st.button("📖 介绍", use_container_width=True,
                        type="primary" if st.session_state.view_mode == "introduction" else "secondary"):
                st.session_state.view_mode = "introduction"
                st.rerun()

        st.divider()

        # View mode selector
        mode_map = {"lesson": 0, "reference": 1, "poems": 2}
        current_mode = st.session_state.view_mode if st.session_state.view_mode in mode_map else "lesson"
        view_mode = st.radio(
            "浏览",
            ["📖 课程", "📚 文法", "🎋 诗歌"],
            index=mode_map.get(current_mode, 0),
            horizontal=False,
        )
        reverse_map = {"📖 课程": "lesson", "📚 文法": "reference", "🎋 诗歌": "poems"}
        new_mode = reverse_map.get(view_mode, "lesson")

        # Track radio selection to detect actual user clicks
        if "last_radio_mode" not in st.session_state:
            st.session_state.last_radio_mode = new_mode

        # Only switch to radio mode if user actually clicked the radio (value changed)
        if new_mode != st.session_state.last_radio_mode:
            st.session_state.view_mode = new_mode
            st.session_state.last_radio_mode = new_mode
        # Also switch if already in a radio-controlled mode and mode matches
        elif st.session_state.view_mode in ["lesson", "reference", "poems"]:
            st.session_state.view_mode = new_mode

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
        st.error("数据库未找到。请先运行构建流程。")
        with st.expander("构建说明"):
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
        st.markdown("## 欢迎来到 WakaWaka")
        st.markdown("从左侧边栏选择一课，开始你的古典日本诗歌学习之旅。")
        return

    nav = st.session_state.navigator
    loader = st.session_state.loader

    # Load lesson content
    lesson = loader.get_lesson_content(lesson_id)
    if not lesson:
        st.error(f"课程未找到：{lesson_id}")
        return

    # Navigation bar
    render_navigation_bar(lesson_id)

    # Get audio paths for poems in this lesson
    audio_files = get_audio_for_lesson(lesson, DATA_DIR)

    # Get poem metadata (author, collection) for display
    from wakawaka.schemas import PoemPresentationStep
    poem_metadata = {}
    for step in lesson.teaching_sequence:
        if isinstance(step, PoemPresentationStep) and step.poem_id:
            poem_data = loader.get_poem(step.poem_id)
            if poem_data:
                poem_metadata[step.poem_id] = {
                    "author": poem_data.author,
                    "collection": poem_data.collection,
                }

    # Inject CSS
    st.markdown(get_vocab_css(), unsafe_allow_html=True)
    st.markdown(get_literary_css(), unsafe_allow_html=True)

    # Get literary loader
    literary_loader = st.session_state.literary_loader

    # Render lesson header
    st.markdown(f'<h1>{lesson.lesson_title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#666;font-style:italic;">{lesson.lesson_summary}</p>', unsafe_allow_html=True)

    # Grammar explanation
    st.markdown(render_grammar_explanation(lesson), unsafe_allow_html=True)

    # Teaching sequence - render step by step to allow inline audio and literary analysis
    for step in lesson.teaching_sequence:
        step_html = render_teaching_step(step, poem_metadata=poem_metadata)
        if step_html:
            st.markdown(step_html, unsafe_allow_html=True)

            # Add audio and literary analysis after poem
            if isinstance(step, PoemPresentationStep) and step.poem_id:
                # Audio player
                if step.poem_id in audio_files:
                    audio_path = audio_files[step.poem_id]
                    st.audio(str(audio_path), format="audio/mp3")

                # Literary analysis panel (collapsible)
                if literary_loader:
                    analysis = literary_loader.get_analysis(step.poem_id)
                    if analysis:
                        st.markdown(render_literary_analysis(analysis), unsafe_allow_html=True)

    # Forward references
    st.markdown(render_forward_references(lesson), unsafe_allow_html=True)

    # Quiz section with text input
    render_quiz_section(lesson)

    # Reference card
    st.divider()
    st.markdown("### 📋 参考卡")
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
            if st.button("← 上一课", use_container_width=True):
                select_lesson(prev_id)

    with col2:
        st.markdown(f"<p style='text-align:center;color:#666;margin-top:0.5em;'>第 {pos} 课 / 共 {total} 课</p>",
                    unsafe_allow_html=True)

    with col3:
        if next_id and nav.is_lesson_available(next_id):
            if st.button("下一课 →", use_container_width=True):
                select_lesson(next_id)


def render_quiz_section(lesson):
    """Render comprehension check with free text input."""
    quizzes = extract_quiz_questions(lesson)
    if not quizzes:
        return

    st.divider()
    st.markdown("### ✍️ 理解检测")

    for quiz in quizzes:
        quiz_key = f"{lesson.lesson_id}_{quiz.index}"

        st.markdown(f"""
        <div class="quiz-section">
            <div class="quiz-header">
                <strong>问题 {quiz.index + 1}</strong>
            </div>
            <div class="quiz-question">{quiz.question}</div>
        </div>
        """, unsafe_allow_html=True)

        # Hint expander
        if quiz.hint:
            with st.expander("💡 显示提示"):
                st.markdown(f"""
                <div class="quiz-hint">{quiz.hint}</div>
                """, unsafe_allow_html=True)

        # User answer input
        user_answer = st.text_area(
            "你的答案：",
            key=f"answer_{quiz_key}",
            placeholder="在揭示答案前写下你的理解...",
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
                if st.button("揭示答案", key=f"reveal_{quiz_key}", type="primary"):
                    st.session_state.quiz_revealed.add(quiz.index)
                    st.rerun()

        if revealed:
            st.markdown(f"""
            <div class="quiz-answer-box">
                <div class="quiz-answer-label">✓ 参考答案</div>
                <div class="quiz-answer-content">{quiz.answer}</div>
            </div>
            """, unsafe_allow_html=True)

            # Self-assessment
            st.markdown("**你掌握了吗？**")
            assess_col1, assess_col2, assess_col3 = st.columns(3)

            # Track assessment in session state
            assessment_key = f"assessment_{quiz_key}"
            if assessment_key not in st.session_state:
                st.session_state[assessment_key] = None

            with assess_col1:
                if st.button("😊 完全掌握", key=f"assess_good_{quiz_key}", use_container_width=True):
                    st.session_state[assessment_key] = "good"
                    st.rerun()
            with assess_col2:
                if st.button("🤔 部分理解", key=f"assess_partial_{quiz_key}", use_container_width=True):
                    st.session_state[assessment_key] = "partial"
                    st.rerun()
            with assess_col3:
                if st.button("😅 需要复习", key=f"assess_review_{quiz_key}", use_container_width=True):
                    st.session_state[assessment_key] = "review"
                    st.rerun()

            # Show feedback based on assessment
            if st.session_state[assessment_key] == "good":
                st.success("太棒了！你已掌握这个概念。")
            elif st.session_state[assessment_key] == "partial":
                st.info("进步不错！建议稍后再复习一下这部分内容。")
            elif st.session_state[assessment_key] == "review":
                st.warning("没关系！慢慢来，再仔细看看这个概念。")

        st.markdown("<br>", unsafe_allow_html=True)


def render_completion_section(lesson_id: str):
    """Render lesson completion section."""
    progress = st.session_state.progress
    nav = st.session_state.navigator
    lesson_progress = progress.get_lesson_progress(lesson_id)

    st.divider()

    if lesson_progress.status == LessonStatus.COMPLETED:
        st.success("✓ 课程已完成！")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("重置进度"):
                progress.reset_lesson(lesson_id)
                st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✓ 标记为已完成", type="primary", use_container_width=True):
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
        st.error("数据库未找到。")
        return

    st.markdown("# 📚 文法参考")

    loader = st.session_state.loader
    progress = st.session_state.progress

    # Search
    search = st.text_input("🔍 搜索文法点", placeholder="例如：助词、动词、助动词")

    grammar_points = loader.get_all_grammar_points()
    if search:
        grammar_points = filter_grammar_points(grammar_points, search)

    st.caption(f"共 {len(grammar_points)} 个文法点")

    # Tabs
    tab1, tab2 = st.tabs(["全部文法", "我的参考卡"])

    with tab1:
        st.markdown(get_reference_css(), unsafe_allow_html=True)
        st.markdown(render_grammar_index(grammar_points), unsafe_allow_html=True)

    with tab2:
        completed_ids = progress.get_completed_lesson_ids()
        if not completed_ids:
            st.info("完成课程以收集参考卡。")
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
        st.error("数据库未找到。")
        return

    st.markdown("# 🎋 诗歌选集")

    # Inject literary CSS
    st.markdown(get_literary_css(), unsafe_allow_html=True)

    loader = st.session_state.loader
    literary_loader = st.session_state.literary_loader
    total_poems = loader.get_poem_count()

    st.caption(f"共收录 {total_poems} 首古典和歌")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        source_filter = st.selectbox("来源", ["全部", "百人一首", "Lapis收藏"])
    with col2:
        difficulty_filter = st.slider("最大难度", 0.0, 1.0, 1.0, 0.1)

    # Load poems
    poems = loader.get_all_poems(limit=100)

    # Map display names to internal values
    source_map = {"全部": "All", "百人一首": "ogura100", "Lapis收藏": "lapis"}
    internal_source = source_map.get(source_filter, "All")

    if internal_source != "All":
        poems = [p for p in poems if p.source == internal_source]
    poems = [p for p in poems if p.difficulty_score <= difficulty_filter]

    st.markdown(f"**显示 {len(poems)} 首诗歌**")

    # Display
    for poem in poems[:30]:
        with st.expander(f"「{poem.text[:25]}…」"):
            st.markdown(f"""
            <div class="poem-anthology-item">
                <div class="poem-anthology-text">{poem.text}</div>
                <div class="poem-anthology-meta">
                    {f"作者：{poem.author}" if poem.author else ""}
                    {f" | 出典：{poem.collection}" if poem.collection else ""}
                    | 难度：{poem.difficulty_score:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if poem.reading_romaji:
                st.caption(f"罗马音：{poem.reading_romaji}")

            # Audio player if available
            audio_path = get_audio_path(poem.id, DATA_DIR)
            if audio_path:
                st.audio(str(audio_path), format="audio/mp3")

            # Literary analysis (expanded by default in anthology)
            if literary_loader:
                analysis = literary_loader.get_analysis(poem.id)
                if analysis:
                    st.markdown(render_literary_analysis(analysis, collapsed=False), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cover Page
# -----------------------------------------------------------------------------

def render_cover_page():
    """Render the welcome cover page in Chinese for target audience."""
    # Get stats if available
    if st.session_state.loader:
        loader = st.session_state.loader
        nav = st.session_state.navigator
        stats = nav.get_progress_summary()
        total_lessons = stats['total_lessons']
        total_poems = loader.get_poem_count()
        total_grammar = loader.get_grammar_point_count()
    else:
        total_lessons = 50
        total_poems = 1000
        total_grammar = 264

    # Title section
    st.markdown("# WakaWaka 🎋")
    st.markdown("### 穿越千年时光的四季诗语")
    st.markdown("""
    *面向汉语母语者的古典日本诗歌学习平台。*
    *以汉字为钥，解锁大和民族的心灵密码。*
    """)

    st.divider()

    # Stats using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="课程", value=total_lessons)
    with col2:
        st.metric(label="和歌", value=total_poems)
    with col3:
        st.metric(label="文法点", value=total_grammar)

    st.divider()

    # Featured poem
    st.markdown("#### 名歌赏析")
    st.markdown("""
    > 秋の田の かりほの庵の 苫をあらみ
    > わが衣手は 露にぬれつつ

    *秋日田边临时草庵，茅屋漏风露水沾衣袖。*

    — 天智天皇（第一首百人一首）
    """)

    st.divider()

    # Features using columns
    st.markdown("#### 为何选择 WakaWaka？")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔤 汉字优势**")
        st.caption("你已认识60%以上的诗歌内容")
    with col2:
        st.markdown("**🎯 语法解码**")
        st.caption("掌握助词与助动词的逻辑")
    with col3:
        st.markdown("**📜 千年诗韵**")
        st.caption("从百人一首到古今和歌集")

    st.divider()

    # Cultural bridge section
    st.markdown("#### 与唐诗宋词的情感共鸣")
    st.markdown("""
    和歌与中国古典诗词共享相同的审美情趣：**物哀**（物の哀れ）——
    对世间万物稍纵即逝的淡淡哀愁与感动。无论是樱花飘落还是秋月当空，
    这份跨越千年的情感，你早已在李白杜甫的诗句中体会过。
    """)

    st.divider()

    # Start buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📖 阅读课程介绍", type="primary", use_container_width=True):
            st.session_state.view_mode = "introduction"
            st.rerun()

        if st.button("⏭️ 直接开始第一课", use_container_width=True):
            st.session_state.view_mode = "lesson"
            if st.session_state.navigator:
                st.session_state.current_lesson_id = st.session_state.navigator.get_first_lesson_id()
            st.rerun()


# -----------------------------------------------------------------------------
# Introduction View
# -----------------------------------------------------------------------------

def render_introduction_view():
    """Render the general introduction lesson using native Streamlit components."""
    intro = st.session_state.introduction

    if not intro:
        st.warning("课程介绍尚未生成。")
        st.markdown("""
        运行以下脚本生成介绍：
        ```bash
        python scripts/06_generate_introduction.py
        ```
        """)
        return

    # Title
    st.markdown(f"# {intro.get('title', '古典日本诗歌入门')}")
    st.markdown(f"*{intro.get('subtitle', '')}*")

    # Sections
    for section in intro.get('sections', []):
        st.markdown(f"## {section.get('heading', '')}")

        content = section.get('content', '')
        # Render content as markdown (Streamlit handles basic markdown natively)
        st.markdown(content)

        # Example poem if present
        if section.get('example_poem'):
            poem = section['example_poem']
            # Get Chinese translation (new field) or fall back to translation
            translation = poem.get('chinese_translation') or poem.get('translation', '')
            st.markdown(f"""
> **{poem.get('text', '')}**

> *{poem.get('romaji', '')}*

> {translation}
            """)

            # Analysis
            if poem.get('analysis'):
                st.info(poem['analysis'])

        # Key points if present
        if section.get('key_points'):
            st.markdown("**要点：**")
            for point in section['key_points']:
                st.markdown(f"- {point}")

    # Closing message
    if intro.get('closing_message'):
        st.divider()
        st.markdown(f"*{intro.get('closing_message')}*")

    # Navigation to first lesson
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("开始学习 →", type="primary", use_container_width=True):
            st.session_state.view_mode = "lesson"
            if st.session_state.navigator:
                st.session_state.current_lesson_id = st.session_state.navigator.get_first_lesson_id()
            st.rerun()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    init_session_state()
    inject_global_css()
    render_sidebar()

    if st.session_state.view_mode == "cover":
        render_cover_page()
    elif st.session_state.view_mode == "introduction":
        render_introduction_view()
    elif st.session_state.view_mode == "lesson":
        render_lesson_view()
    elif st.session_state.view_mode == "reference":
        render_reference_view()
    elif st.session_state.view_mode == "poems":
        render_poems_view()


if __name__ == "__main__":
    main()

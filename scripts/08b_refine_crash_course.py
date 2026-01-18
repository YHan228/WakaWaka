#!/usr/bin/env python3
"""
Synthesize Crash Course from Multiple Trials.

This script takes multiple crash course trials and synthesizes them into
one optimal course by combining the best elements from each trial.

Usage:
    python scripts/08b_refine_crash_course.py --input-dir data/crash_course_trials
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from wakawaka.schemas.crash_course import CrashCourse

load_dotenv()


SYNTHESIS_PROMPT = """你是资深日语教学课程编辑，负责将多个独立的课程草稿综合成一个最优版本。

## 项目背景：WakaWaka 古典和歌学习平台

WakaWaka 是一个面向汉语母语者的古典日本诗歌（和歌）学习平台。主课程包含50多节课，教授古典日语语法和诗歌阅读。

**这个速成课程是入口课程**，帮助零基础学习者：
1. 掌握平假名，能够拼读日语
2. 理解基本语法概念（助词、语序、动词）
3. 建立信心，准备好进入主课程

完成速成课程后，学习者将进入主课程，学习：
- 古典日语助动词（き、けり、つ、ぬ等）
- 和歌修辞手法（枕词、序词、掛詞）
- 百人一首等经典诗歌

## 你的任务

你将收到 {num_trials} 份独立生成的日语速成课程草稿。请综合出一份最优的课程。

## 综合原则

### 1. 内容完整性（必须）
- **平假名完整覆盖46个**：
  - あ行(5) + か行(5) + さ行(5) + た行(5) + な行(5) + は行(5) + ま行(5) + や行(3) + ら行(5) + わ行(2) + ん(1) = 46
- **必须覆盖的语法**：音读/训读、SOV语序、五大助词（の、は、が、を、に）、动词基本形态
- **桥接内容**：古典日语与现代日语的区别提示，为主课程做铺垫

### 2. 日语发音特点（重要）
**必须介绍日语的声调系统（pitch accent/高低アクセント）**：
- 日语不是声调语言（不像中文有四声），而是有高低音节模式
- 例如：箸（はし）筷子 vs 橋（はし）桥 vs 端（はし）边缘 - 音节相同但音高模式不同
- 用简单的高低标记法介绍：如 はし↓（筷子：高-低）vs は↑し（桥：低-高）
- 这对初学者理解日语发音非常重要

### 3. 教学质量（严格要求）
- **概念解释必须充分**：不要只列出知识点，要有详细的解释和多个例子
- **避免空洞内容**：每个步骤都要有实质内容，不要只是"接下来我们学习X"
- **例句要有分析**：不要只给例句，要逐词解释
- **避免假大空的结语**：不要写"祝学习愉快"这种废话，最后一课要有实质的和歌分析

### 4. 练习设计
- 练习必须是视觉性的（不依赖音频）
- 练习格式：
  - 匹配题：{{"left": "日语", "right": "中文"}}
  - 选择题：{{"question": "问题", "options": ["A", "B", "C"], "answer": "正确答案", "explanation": "解释"}}
  - 填空题：{{"sentence": "句子___", "blank": "答案"}}
- 每课至少1个练习

### 5. 语言质量
- 使用准确的中文表达，避免"确实"这种错别字（应该是"缺失"）
- 专业术语要准确：pitch accent（声调/音高重音）、音读、训读等
- 不要使用Markdown格式（如**粗体**），直接用HTML格式或普通文本

### 6. 课程结构
- 课程数量根据内容合理安排（通常6-8课）
- 课程标题要一致（不要"上"没有"下"）
- 每课时长15-25分钟为宜
- **最后一课必须是完整的和歌分析实战**，不是简单的结语

### 7. 综合策略
- 比较各草稿中相同主题的内容，选择解释最清晰、例子最好的版本
- 整合各草稿的独特优点
- 保持整体风格一致
- 去除冗余，但不要省略重要解释

## 草稿内容

{drafts}

## 输出要求

返回综合后的完整课程JSON，格式如下：

```json
{{
  "title": "课程标题",
  "subtitle": "副标题",
  "description": "课程描述",
  "target_audience": "零基础汉语母语者",
  "lessons": [
    {{
      "lesson_id": "crash_01",
      "lesson_number": 1,
      "title": "课程标题",
      "subtitle": "副标题",
      "estimated_minutes": 15,
      "objectives": ["目标1", "目标2"],
      "teaching_sequence": [...],
      "comprehension_check": {{...}}
    }}
  ],
  "version": "1.0",
  "generated_at": "ISO时间戳"
}}
```

重要：
1. 仅返回JSON，不要其他文字
2. 确保所有课程都包含在输出中
3. 确保平假名完整覆盖46个
4. 确保练习不依赖音频
"""


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()

    # Try direct parse
    if text.startswith('{'):
        text = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        text = json_match.group(1).strip()
    else:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)

    text = re.sub(r',\s*([}\]])', r'\1', text)

    return json.loads(text)


def load_trials(input_dir: Path) -> list[dict]:
    """Load all trial JSON files from directory."""
    trials = []
    for trial_path in sorted(input_dir.glob("trial_*.json")):
        try:
            with open(trial_path, encoding="utf-8") as f:
                data = json.load(f)
            trials.append({
                "path": trial_path.name,
                "data": data,
            })
        except Exception as e:
            print(f"Warning: Failed to load {trial_path}: {e}")
    return trials


def format_trials_for_prompt(trials: list[dict]) -> str:
    """Format trials as text for the synthesis prompt."""
    parts = []
    for i, trial in enumerate(trials, 1):
        data = trial["data"]
        lessons_summary = []
        for lesson in data.get("lessons", []):
            lessons_summary.append(f"    {lesson.get('lesson_number', '?')}. {lesson.get('title', 'N/A')}")

        parts.append(f"""
=== 草稿 {i} ({trial['path']}) ===
标题: {data.get('title', 'N/A')}
课程数: {len(data.get('lessons', []))}
课程列表:
{chr(10).join(lessons_summary)}

完整JSON:
```json
{json.dumps(data, indent=2, ensure_ascii=False)}
```
""")
    return "\n".join(parts)


def count_kana_coverage(course: dict) -> dict:
    """Count hiragana coverage in the course."""
    all_kana = set()
    for lesson in course.get("lessons", []):
        for step in lesson.get("teaching_sequence", []):
            if step.get("type") == "kana_chart":
                rows = step.get("rows", [])
                for row in rows:
                    # Handle both flat list and nested list formats
                    if isinstance(row, dict):
                        # Flat format: rows is list of cell dicts
                        if row.get("kana"):
                            all_kana.add(row["kana"])
                    elif isinstance(row, list):
                        # Nested format: rows is list of lists of cells
                        for cell in row:
                            if cell:
                                if isinstance(cell, dict) and cell.get("kana"):
                                    all_kana.add(cell["kana"])
                                elif isinstance(cell, str) and len(cell) == 1:
                                    all_kana.add(cell)
                    elif isinstance(row, str) and len(row) == 1:
                        all_kana.add(row)
    return {
        "count": len(all_kana),
        "kana": sorted(all_kana),
    }


def main():
    parser = argparse.ArgumentParser(description="Synthesize crash course from trials")
    parser.add_argument("--input-dir", default="data/crash_course_trials", help="Directory with trial files")
    parser.add_argument("--output", default="data/crash_course.json", help="Output file")
    parser.add_argument("--model", default="gemini-3-pro-preview", help="Model to use")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load trials
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        print("Run the generation script first:")
        print("  python scripts/08_generate_crash_course.py --trials 10")
        sys.exit(1)

    print(f"Loading trials from {input_dir}...")
    trials = load_trials(input_dir)

    if not trials:
        print("Error: No trials found")
        sys.exit(1)

    print(f"Loaded {len(trials)} trials:")
    for trial in trials:
        data = trial["data"]
        kana = count_kana_coverage(data)
        print(f"  {trial['path']}: {len(data.get('lessons', []))} lessons, {kana['count']} kana")

    # Format prompt
    drafts_text = format_trials_for_prompt(trials)
    prompt = SYNTHESIS_PROMPT.format(
        num_trials=len(trials),
        drafts=drafts_text,
    )

    print(f"\nSynthesizing with {args.model}...")
    print("This may take a few minutes...")

    try:
        response = client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for synthesis
                max_output_tokens=100000,
            ),
        )

        response_text = response.text
        if not response_text and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text

        if not response_text:
            print("Error: Empty response from model")
            sys.exit(1)

        # Parse result
        result = extract_json(response_text)

        # Add metadata
        result["generated_at"] = datetime.now().isoformat()
        result["synthesis_info"] = {
            "num_trials": len(trials),
            "model": args.model,
        }

        # Validate
        print("\nValidating synthesized course...")
        try:
            course = CrashCourse(**result)
            print(f"✓ Validation passed")
        except Exception as e:
            print(f"Warning: Validation failed: {e}")
            print("Saving anyway...")

        # Check kana coverage
        kana = count_kana_coverage(result)
        print(f"Hiragana coverage: {kana['count']}/46")
        if kana['count'] < 46:
            print(f"Warning: Missing some hiragana!")

        # Print summary
        print(f"\nSynthesized course: {result.get('title', 'N/A')}")
        print(f"Lessons: {len(result.get('lessons', []))}")
        total_time = sum(l.get("estimated_minutes", 0) for l in result.get("lessons", []))
        print(f"Total time: ~{total_time} minutes")
        for lesson in result.get("lessons", []):
            print(f"  {lesson.get('lesson_number', '?')}. {lesson.get('title', 'N/A')} ({len(lesson.get('teaching_sequence', []))} steps)")

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Synthesized course saved to {output_path}")

    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

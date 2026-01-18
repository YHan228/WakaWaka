#!/usr/bin/env python3
"""
Generate Japanese Crash Course for Absolute Beginners - Ensemble Method.

This script generates multiple independent trials of the crash course,
each with high temperature for diversity. The trials are then synthesized
by a separate script (08b_refine_crash_course.py).

Usage:
    python scripts/08_generate_crash_course.py --trials 10 --parallel 5
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


CRASH_COURSE_PROMPT = """你是日语教学专家，专门为零基础汉语母语者设计入门课程。

## 项目背景
这是 WakaWaka 古典和歌学习平台的入口课程。学完这个速成课程后，学习者将进入主课程学习古典日语语法和诗歌阅读。

## 学生背景
- 汉语母语者，能流利阅读汉字
- 完全没有日语基础（零基础）
- 目标是学习阅读古典日本诗歌（和歌）
- 成年学习者，有学习语言的动机

## 教学原则
1. **从已知到未知**：充分利用汉字认知优势，从熟悉的汉字开始
2. **完整覆盖平假名**：必须教授全部46个基本平假名
3. **循序渐进**：每课15-25分钟，目标明确
4. **解释充分**：每个概念都要有详细解释和多个例子，不要只列知识点
5. **实践为主**：每课包含练习（不依赖音频，纯视觉练习）
6. **桥接古典**：为古典日语（和歌）学习做准备

## 必须覆盖的内容

### 平假名（必须完整覆盖46个）
- あ行：あ い う え お（元音）
- か行：か き く け こ
- さ行：さ し す せ そ
- た行：た ち つ て と
- な行：な に ぬ ね の
- は行：は ひ ふ へ ほ
- ま行：ま み む め も
- や行：や ゆ よ
- ら行：ら り る れ ろ
- わ行：わ を
- ん

### 日语发音特点（重要！）
**必须介绍日语的声调系统（pitch accent/高低アクセント）**：
- 日语不是像中文那样的声调语言，而是有高低音节模式（pitch accent）
- 同音词靠音高区分：箸（はし↓）筷子 vs 橋（は↑し）桥
- 用简单标记法表示：↑表示升调，↓表示降调
- 这对理解日语发音非常重要，汉语母语者常忽略这点

### 语法基础
- 汉字的音读和训读（要有充分解释和例子）
- SOV语序（主语-宾语-动词）- 要对比中文SVO
- 核心助词：の、は、が、を、に（每个都要详细解释用法）
- 动词基础形态和时态
- 古典日语与现代日语的区别提示（为主课程铺垫）

### 最终目标
能够初步解读一首简单的和歌（最后一课必须包含完整的和歌分析，不是泛泛的结语）

## 课程设计要求

1. **课程数量**：根据内容合理安排，通常6-10课
2. **每课结构**：
   - 明确的学习目标
   - 渐进式教学步骤
   - 至少1个练习（不依赖音频）
   - 本课要点总结

3. **步骤类型**（teaching_sequence中使用）：
   - `text`: 文字说明 {{"type": "text", "title": "标题", "content": "内容（不要用**markdown**，用普通文本）"}}
   - `kana_chart`: 假名表 {{"type": "kana_chart", "title": "标题", "chart_type": "hiragana", "rows": [[{{"kana": "あ", "romaji": "a", "mnemonic": "记忆技巧", "pitch": "低"}}]], "highlight_row": 0}}
   - `vocab_intro`: 词汇介绍（含声调标记）{{"type": "vocab_intro", "title": "标题", "items": [{{"japanese": "桃", "reading": "もも↓", "chinese": "桃", "meaning": "桃子", "note": "高-低调"}}]}}
   - `practice`: 练习 {{"type": "practice", "title": "标题", "instruction": "说明", "exercise_type": "match|select|fill_blank", "items": [...]}}
   - `example_sentence`: 例句分析 {{"type": "example_sentence", "sentence": "句子", "reading": "读音", "translation": "翻译", "breakdown": [{{"part": "词", "role": "角色", "meaning": "意思"}}]}}
   - `mini_poem`: 诗歌展示（最后一课必须有详细分析）{{"type": "mini_poem", "poem_text": "诗文", "reading": "读音", "translation": "翻译", "analysis": "详细的逐句分析"}}
   - `summary`: 总结 {{"type": "summary", "title": "本课要点", "key_points": ["要点1", "要点2"], "preview_next": "下课预告（最后一课不要写空话）"}}
   - `tip`: 提示 {{"type": "tip", "tip_type": "learning|cultural|memory|comparison", "content": "内容"}}

4. **练习格式**（items数组）：
   - 匹配题：{{"left": "日语", "right": "中文"}}
   - 选择题：{{"question": "问题", "options": ["A", "B", "C"], "answer": "正确答案", "explanation": "解释"}}
   - 填空题：{{"sentence": "句子___", "blank": "答案"}}

## 输出格式

返回完整的JSON课程：

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
      "subtitle": "吸引人的副标题",
      "estimated_minutes": 15,
      "objectives": ["目标1", "目标2"],
      "teaching_sequence": [...],
      "comprehension_check": {{
        "questions": [
          {{"question": "问题", "question_type": "multiple_choice", "options": ["A", "B", "C"], "correct_answer": "A", "explanation": "解释"}}
        ]
      }}
    }}
  ],
  "version": "1.0"
}}
```

重要：
1. 仅返回JSON，不要其他文字
2. 确保平假名完整覆盖
3. 练习不要依赖音频
4. 课程标题要一致（不要"上"没有"下"）
"""


def extract_json(text: str, debug: bool = False) -> dict:
    """Extract JSON from LLM response."""
    if debug:
        print(f"  [DEBUG] Response length: {len(text)}")

    text = text.strip()

    # Try direct parse first
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

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Text preview: {text[:500]}...")
        raise


def generate_trial(
    client: genai.Client,
    model_name: str,
    trial_id: int,
    temperature: float = 0.95,
    debug: bool = False,
) -> dict:
    """Generate a single trial of the crash course."""
    print(f"  Trial {trial_id}: Starting (temp={temperature})...")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=CRASH_COURSE_PROMPT,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=60000,
            ),
        )

        response_text = response.text
        if not response_text and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text

        if not response_text:
            raise ValueError("Empty response from model")

        result = extract_json(response_text, debug=debug)

        # Basic validation
        lessons = result.get("lessons", [])
        print(f"  Trial {trial_id}: Complete ({len(lessons)} lessons)")

        return {"trial_id": trial_id, "result": result, "success": True}

    except Exception as e:
        print(f"  Trial {trial_id}: Failed - {e}")
        return {"trial_id": trial_id, "error": str(e), "success": False}


def main():
    parser = argparse.ArgumentParser(description="Generate crash course trials")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials (default: 10)")
    parser.add_argument("--parallel", type=int, default=5, help="Parallel workers (default: 5)")
    parser.add_argument("--model", default="gemini-3-pro-preview", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.95, help="Temperature (default: 0.95)")
    parser.add_argument("--output-dir", default="data/crash_course_trials", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"Generating {args.trials} crash course trials")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Parallel workers: {args.parallel}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate trials in parallel
    successful_trials = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                generate_trial, client, args.model, i, args.temperature, args.debug
            ): i
            for i in range(1, args.trials + 1)
        }

        for future in as_completed(futures):
            trial_id = futures[future]
            try:
                result = future.result()
                if result["success"]:
                    successful_trials.append(result)
                    # Save individual trial
                    trial_path = output_dir / f"trial_{trial_id:02d}.json"
                    with open(trial_path, "w", encoding="utf-8") as f:
                        json.dump(result["result"], f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Trial {trial_id}: Exception - {e}")

    print(f"\nSuccessful trials: {len(successful_trials)}/{args.trials}")

    if not successful_trials:
        print("Error: No trials generated successfully")
        sys.exit(1)

    # Save summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "temperature": args.temperature,
        "total_trials": args.trials,
        "successful_trials": len(successful_trials),
        "trials": [
            {
                "trial_id": t["trial_id"],
                "lessons": len(t["result"].get("lessons", [])),
                "title": t["result"].get("title", ""),
            }
            for t in sorted(successful_trials, key=lambda x: x["trial_id"])
        ],
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Trials saved to: {output_dir}")
    print(f"  Summary: {summary_path}")

    # Show trial overview
    print("\nTrial overview:")
    for t in sorted(successful_trials, key=lambda x: x["trial_id"]):
        lessons = t["result"].get("lessons", [])
        print(f"  Trial {t['trial_id']}: {len(lessons)} lessons - {t['result'].get('title', 'N/A')}")

    print(f"\nNext step: Run synthesis script")
    print(f"  python scripts/08b_refine_crash_course.py --input-dir {output_dir}")


if __name__ == "__main__":
    main()

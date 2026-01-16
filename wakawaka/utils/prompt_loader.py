"""
Prompt loader utility for WakaDecoder.

Loads YAML prompt templates from the prompts/ directory.
"""

from pathlib import Path
from typing import Any
import yaml


# Default prompts directory (relative to project root)
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(name: str, prompts_dir: Path | None = None) -> dict[str, Any]:
    """
    Load a prompt template by name.

    Args:
        name: Prompt name without .yaml extension (e.g., "annotate")
        prompts_dir: Optional custom prompts directory

    Returns:
        Dict containing the parsed YAML prompt template with keys:
        - meta: version, model, temperature
        - system: system prompt string
        - user_template: user prompt template with {placeholders}
        - validation: optional validation rules

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    dir_path = prompts_dir or PROMPTS_DIR
    file_path = dir_path / f"{name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with provided values.

    Args:
        template: Template string with {placeholders}
        **kwargs: Values to substitute

    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)


def get_available_prompts(prompts_dir: Path | None = None) -> list[str]:
    """
    List all available prompt templates.

    Args:
        prompts_dir: Optional custom prompts directory

    Returns:
        List of prompt names (without .yaml extension)
    """
    dir_path = prompts_dir or PROMPTS_DIR
    if not dir_path.exists():
        return []
    return [p.stem for p in dir_path.glob("*.yaml")]

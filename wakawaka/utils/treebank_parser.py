"""
Penn Treebank parser for ONCOJ corpus.

Parses bracketed tree format and extracts text from PHON (phonetic) nodes.
"""

import re
from dataclasses import dataclass
from typing import Iterator


@dataclass
class ParsedText:
    """A parsed text from ONCOJ corpus."""
    text_id: str
    text: str
    source_file: str
    metadata: dict


# Map collection codes to full names
COLLECTION_NAMES = {
    'MYS': '万葉集',
    'KK': '古事記',
    'NSK': '日本書紀',
    'FK': '風土記',
    'SM': '正倉院文書',
    'BS': '仏足石歌',
    'JSHT': '上代特殊仮名遣',
}


def extract_phon_text(tree_str: str) -> str:
    """
    Extract text from PHON (phonetic) nodes in ONCOJ format.

    ONCOJ uses (PHON xxx) to mark the phonetic reading of each morpheme.

    Example:
        (N (L050877 (PHON ato))) -> 'ato'

    Returns concatenated phonetic text.
    """
    # Pattern to match (PHON xxx) nodes
    phon_pattern = r'\(PHON\s+([^)]+)\)'

    phonemes = []
    for match in re.finditer(phon_pattern, tree_str):
        phon = match.group(1).strip()
        if phon:
            phonemes.append(phon)

    return ''.join(phonemes)


def extract_text_id(tree_str: str) -> str | None:
    """
    Extract the ID from an ONCOJ tree.

    ONCOJ format: (ID BS.1) or (ID MYS.1.1) at the end of each tree.
    """
    id_pattern = r'\(ID\s+([^)]+)\)'
    match = re.search(id_pattern, tree_str)
    if match:
        return match.group(1).strip()
    return None


def parse_text_id(text_id: str) -> dict:
    """
    Parse ONCOJ text ID into metadata.

    Examples:
        MYS.1.1 -> {'collection': 'MYS', 'book': '1', 'poem': '1'}
        KK.1 -> {'collection': 'KK', 'poem': '1'}
        BS.1 -> {'collection': 'BS', 'poem': '1'}
    """
    parts = text_id.split('.')
    metadata = {'raw_id': text_id}

    if parts:
        metadata['collection'] = parts[0]
        if len(parts) >= 2:
            metadata['book'] = parts[1]
        if len(parts) >= 3:
            metadata['poem'] = parts[2]

    if metadata.get('collection') in COLLECTION_NAMES:
        metadata['collection_name'] = COLLECTION_NAMES[metadata['collection']]

    return metadata


def parse_oncoj_file(content: str, source_file: str) -> Iterator[ParsedText]:
    """
    Parse an ONCOJ .psd file containing multiple bracketed trees.

    ONCOJ format:
    - Each poem is wrapped in outer parentheses
    - Phonetic content is in (PHON xxx) nodes
    - ID is at the end: (ID MYS.1.1)
    - Trees are separated by blank lines

    Args:
        content: File content as string
        source_file: Source filename for provenance

    Yields:
        ParsedText objects
    """
    # Split into individual trees by finding balanced outer parentheses
    # Each tree starts with ( ( and ends with ))
    trees = []
    depth = 0
    current_tree = []

    for char in content:
        if char == '(':
            depth += 1
            current_tree.append(char)
        elif char == ')':
            current_tree.append(char)
            depth -= 1
            if depth == 0 and current_tree:
                tree_str = ''.join(current_tree).strip()
                if tree_str:
                    trees.append(tree_str)
                current_tree = []
        elif depth > 0:
            current_tree.append(char)

    for tree_str in trees:
        # Extract ID
        text_id = extract_text_id(tree_str)
        if not text_id:
            continue

        # Extract phonetic text
        text = extract_phon_text(tree_str)
        if not text or len(text) < 5:  # Skip very short fragments
            continue

        # Parse metadata
        metadata = parse_text_id(text_id)

        yield ParsedText(
            text_id=text_id,
            text=text,
            source_file=source_file,
            metadata=metadata
        )


def parse_simple_bracketed(content: str, source_file: str) -> Iterator[ParsedText]:
    """
    Fallback parser for simpler bracketed formats.

    Tries to extract any readable text from bracketed structures.
    """
    depth = 0
    current_tree = []
    tree_id = 0

    for char in content:
        if char == '(':
            depth += 1
            current_tree.append(char)
        elif char == ')':
            current_tree.append(char)
            depth -= 1
            if depth == 0 and current_tree:
                tree_str = ''.join(current_tree)

                # Try PHON extraction first
                text = extract_phon_text(tree_str)

                # Fallback to simple leaf extraction
                if not text:
                    leaf_pattern = r'\(([A-Z0-9_-]+)\s+([^()]+?)\)'
                    leaves = []
                    for match in re.finditer(leaf_pattern, tree_str):
                        _, word = match.groups()
                        word = word.strip()
                        if word and not word.startswith('*') and word not in ('0', '*T*', '*PRO*'):
                            leaves.append(word)
                    text = ''.join(leaves)

                if text and len(text) >= 5:
                    tree_id += 1
                    text_id = extract_text_id(tree_str) or f"{source_file}_{tree_id}"
                    metadata = parse_text_id(text_id) if '.' in text_id else {}

                    yield ParsedText(
                        text_id=text_id,
                        text=text,
                        source_file=source_file,
                        metadata=metadata
                    )
                current_tree = []
        elif depth > 0:
            current_tree.append(char)

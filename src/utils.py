# src/utils.py
import os
from typing import List


def list_text_files(folder: str) -> List[str]:
    """List all .txt files in a folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]


def read_text(path: str) -> str:
    """Read text file content."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

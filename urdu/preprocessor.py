"""
AenPi.urdu.preprocessor
------------------------
Urdu text preprocessing utilities.
Handles tokenization, punctuation removal, and full preprocessing pipeline.
"""

import re

# ─── Urdu Unicode range ───────────────────────────────────────────────────────
URDU_PUNCTUATION = r'[؟،۔!?,.\"\'\(\)\[\]{}<>؛:۱۲۳۴۵۶۷۸۹۰]'
ARABIC_DIACRITICS = r'[\u0610-\u061A\u064B-\u065F]'   # harakat / tashkeel
SPECIAL_CHARS     = r'[^\u0600-\u06FF\u0750-\u077F\s]'  # non-Urdu, non-space


def remove_punctuation(text: str) -> str:
    """Remove Urdu and common punctuation from text."""
    text = re.sub(URDU_PUNCTUATION, ' ', text)
    return text


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel) from text."""
    return re.sub(ARABIC_DIACRITICS, '', text)


def remove_non_urdu(text: str) -> str:
    """Remove characters outside the Urdu/Arabic Unicode block."""
    return re.sub(SPECIAL_CHARS, ' ', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> list:
    """
    Split Urdu text into word tokens.

    Args:
        text (str): Raw or preprocessed Urdu string.

    Returns:
        list: List of word tokens.

    Example:
        >>> tokenize("میرا نام احمد ہے")
        ['میرا', 'نام', 'احمد', 'ہے']
    """
    return text.split()


def preprocess(text: str, remove_diac: bool = True, remove_non_urdu_chars: bool = True) -> str:
    """
    Full preprocessing pipeline for Urdu text.

    Steps:
        1. Remove diacritics (optional)
        2. Remove non-Urdu characters (optional)
        3. Remove punctuation
        4. Normalize whitespace

    Args:
        text (str): Raw Urdu text.
        remove_diac (bool): Whether to strip tashkeel. Default True.
        remove_non_urdu_chars (bool): Strip non-Urdu characters. Default True.

    Returns:
        str: Cleaned Urdu text.

    Example:
        >>> preprocess("میرا نام احمد ہے!")
        'میرا نام احمد ہے'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    if remove_diac:
        text = remove_diacritics(text)
    if remove_non_urdu_chars:
        text = remove_non_urdu(text)

    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text

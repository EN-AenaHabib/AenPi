"""
AenPi.urdu.normalizer
----------------------
Urdu text normalization:
  - Fixes common character-level variants (alef, ya, kaf, ha)
  - Removes ligature forms
  - Standardizes spacing around Urdu punctuation
"""

import re

# ─── Character normalization maps ─────────────────────────────────────────────
# Alef variants  → bare alef ا
ALEF_VARIANTS = str.maketrans({
    '\u0622': '\u0627',  # آ  → ا
    '\u0623': '\u0627',  # أ  → ا
    '\u0625': '\u0627',  # إ  → ا
    '\u0671': '\u0627',  # ٱ  → ا
    '\u0672': '\u0627',  # ٲ  → ا
    '\u0673': '\u0627',  # ٳ  → ا
})

# Ya variants → ی (Urdu ya)
YA_VARIANTS = str.maketrans({
    '\u0649': '\u06CC',  # ى (Arabic alef maqsura) → ی
    '\u064A': '\u06CC',  # ي (Arabic ya)           → ی
})

# Kaf variants → ک (Urdu kaf)
KAF_VARIANTS = str.maketrans({
    '\u0643': '\u06A9',  # ك (Arabic kaf) → ک
})

# Ha variants → ہ (Urdu he)
HA_VARIANTS = str.maketrans({
    '\u0647': '\u06C1',  # ه (Arabic he) → ہ
})

# Ligature forms (presentation forms → remove / replace with space)
# These appear in some older Urdu encodings
LIGATURE_PATTERN = re.compile(r'[\uFB50-\uFDFF\uFE70-\uFEFF]')


def normalize_alef(text: str) -> str:
    """Normalize all Alef variants to bare Alef ا."""
    return text.translate(ALEF_VARIANTS)


def normalize_ya(text: str) -> str:
    """Normalize Arabic Ya variants to Urdu Ya ی."""
    return text.translate(YA_VARIANTS)


def normalize_kaf(text: str) -> str:
    """Normalize Arabic Kaf ك to Urdu Kaf ک."""
    return text.translate(KAF_VARIANTS)


def normalize_ha(text: str) -> str:
    """Normalize Arabic Ha ه to Urdu He ہ."""
    return text.translate(HA_VARIANTS)


def remove_ligatures(text: str) -> str:
    """Remove or strip Arabic presentation-form ligatures."""
    return LIGATURE_PATTERN.sub(' ', text)


def normalize_punctuation_spacing(text: str) -> str:
    """Ensure single space after Urdu punctuation marks."""
    text = re.sub(r'([۔،؟])\s*', r'\1 ', text)
    return text.strip()


def normalize(text: str) -> str:
    """
    Full normalization pipeline for Urdu text.

    Steps:
        1. Normalize Alef variants
        2. Normalize Ya variants
        3. Normalize Kaf variants
        4. Normalize Ha variants
        5. Remove ligature forms
        6. Fix punctuation spacing
        7. Collapse extra whitespace

    Args:
        text (str): Raw Urdu text.

    Returns:
        str: Normalized Urdu text.

    Example:
        >>> normalize("ﻣﯿﺮﺍ ﻧﺎﻡ احمد ہے")
        'میرا نام احمد ہے'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    text = normalize_alef(text)
    text = normalize_ya(text)
    text = normalize_kaf(text)
    text = normalize_ha(text)
    text = remove_ligatures(text)
    text = normalize_punctuation_spacing(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

"""
AenPi.urdu.stopwords
---------------------
Urdu stopword list and removal utility.
"""

# ─── Core Urdu stopwords ──────────────────────────────────────────────────────
URDU_STOPWORDS = {
    "یہ", "وہ", "ہے", "ہیں", "تھا", "تھی", "تھے",
    "اور", "یا", "لیکن", "کہ", "جو", "جب", "اگر",
    "تو", "بھی", "نہیں", "نہ", "ہاں", "کیا", "کیوں",
    "کب", "کہاں", "کیسے", "کون", "کس", "کسی", "کوئی",
    "سب", "کچھ", "اس", "اُس", "اِس", "اِن", "اُن",
    "میں", "تم", "ہم", "آپ", "وے", "انہوں", "انہیں",
    "مجھے", "تمہیں", "ہمیں", "آپ کو", "اسے", "اُسے",
    "میرا", "تمہارا", "ہمارا", "آپ کا", "اس کا", "اُن کا",
    "پر", "میں", "سے", "تک", "کے", "کی", "کو", "نے",
    "کے لیے", "کے ساتھ", "کے بعد", "کے پہلے",
    "ہو", "ہوا", "ہوئی", "ہوئے", "ہوتا", "ہوتی", "ہوتے",
    "کر", "کرنا", "کیا", "کی", "کرتا", "کرتی", "کرتے",
    "جا", "جانا", "گیا", "گئی", "گئے",
    "آ", "آنا", "آیا", "آئی", "آئے",
    "ایک", "دو", "تین", "چار", "پانچ",
    "اب", "پھر", "پہلے", "بعد", "ابھی",
    "بہت", "زیادہ", "کم", "اچھا", "برا",
    "ہاں", "نہیں", "شاید", "ضرور",
    "جی", "صاحب", "بھائی",
}


def remove_stopwords(text: str, custom_stopwords: set = None) -> str:
    """
    Remove Urdu stopwords from tokenized text.

    Args:
        text (str): Whitespace-separated Urdu text.
        custom_stopwords (set): Optional extra stopwords to remove.

    Returns:
        str: Text with stopwords removed.

    Example:
        >>> remove_stopwords("یہ ایک اچھا دن ہے")
        'دن'
    """
    stopwords = URDU_STOPWORDS.copy()
    if custom_stopwords:
        stopwords.update(custom_stopwords)

    tokens = text.split()
    filtered = [token for token in tokens if token not in stopwords]
    return ' '.join(filtered)


def get_stopwords() -> set:
    """Return a copy of the default Urdu stopword set."""
    return URDU_STOPWORDS.copy()

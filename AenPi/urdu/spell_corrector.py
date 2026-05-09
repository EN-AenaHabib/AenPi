"""
AenPi.urdu.spell_corrector
---------------------------
Lightweight Urdu spell correction using Levenshtein edit distance.

Vocabulary priority:
  1. Auto-downloads 150,000+ word list from urduhack/urdu-words (on first use)
  2. Caches it locally so it only downloads ONCE
  3. Falls back to built-in ~200 words if no internet available
"""

import os
import json

# ─── Cache config ─────────────────────────────────────────────────────────────
_CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".aenpi_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "urdu_vocab.json")
_VOCAB_URL  = "https://raw.githubusercontent.com/urduhack/urdu-words/master/words.txt"

# ─── Built-in fallback vocabulary (~200 words) ───────────────────────────────
_FALLBACK_VOCABULARY = {
    "نام", "گھر", "ملک", "شہر", "گاؤں", "دن", "رات", "وقت", "کام", "بات",
    "جگہ", "راستہ", "دروازہ", "کمرہ", "کتاب", "قلم", "میز", "کرسی", "پانی",
    "کھانا", "روٹی", "چائے", "دودھ", "پھل", "سبزی", "بازار", "دکان", "مکان",
    "درخت", "پھول", "آسمان", "زمین", "سورج", "چاند", "ستارہ", "بادل",
    "ہوا", "بارش", "دریا", "سمندر", "پہاڑ", "جنگل", "باغ", "کھیت", "سڑک",
    "گاڑی", "بس", "ٹرین", "اسکول", "کالج", "یونیورسٹی", "ہسپتال", "مسجد",
    "دفتر", "فیکٹری", "آدمی", "عورت", "لڑکا", "لڑکی", "بچہ", "بچی", "بچے",
    "ماں", "باپ", "بھائی", "بہن", "دادا", "دادی", "نانا", "نانی", "دوست",
    "استاد", "شاگرد", "ڈاکٹر", "انجینئر", "وکیل", "کسان", "مزدور",
    "احمد", "محمد", "علی", "حسن", "عمر", "عثمان", "بلال", "زید",
    "فاطمہ", "عائشہ", "زینب", "مریم", "سارہ", "نور", "حنا",
    "پاکستان", "لاہور", "کراچی", "اسلام آباد", "پشاور", "کوئٹہ", "ملتان",
    "فیصل آباد", "راولپنڈی", "حیدرآباد", "ہندوستان", "ایران", "افغانستان",
    "کرنا", "جانا", "آنا", "کھانا", "پینا", "سونا", "اٹھنا", "بیٹھنا",
    "چلنا", "دوڑنا", "پڑھنا", "لکھنا", "بولنا", "سننا", "دیکھنا", "سمجھنا",
    "جاننا", "دینا", "لینا", "رکھنا", "ملنا", "بتانا", "سیکھنا", "کھیلنا",
    "ہے", "ہیں", "ہوں", "تھا", "تھی", "تھے", "ہوگا", "ہوگی",
    "کیا", "کی", "کیے", "کرتا", "کرتی", "کرتے", "گیا", "گئی", "گئے",
    "آیا", "آئی", "آئے", "رہا", "رہی", "رہے",
    "اچھا", "برا", "بڑا", "چھوٹا", "لمبا", "موٹا", "پتلا", "خوبصورت",
    "مشہور", "نیا", "پرانا", "تیز", "سست", "ہوشیار", "امیر", "غریب",
    "خوش", "ناخوش", "صحیح", "غلط", "آسان", "مشکل", "گرم", "ٹھنڈا",
    "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو", "دس",
    "بیس", "تیس", "چالیس", "پچاس", "سو", "ہزار", "لاکھ", "کروڑ",
    "آج", "کل", "پرسوں", "صبح", "دوپہر", "شام", "رات", "ابھی", "پھر",
    "پہلے", "بعد", "جلدی", "دیر", "ہمیشہ", "کبھی",
    "بہت", "کم", "زیادہ", "تھوڑا", "سب", "کچھ", "سارا", "پورا",
    "بھی", "تو", "ہی", "نہیں", "نہ", "ہاں", "جی", "شکریہ",
    "اللہ", "انسان", "دنیا", "زندگی", "موت", "محبت", "نفرت",
    "خوشی", "غم", "امید", "خوف", "سچ", "جھوٹ", "عزت", "محنت", "قسمت",
}

# ─── Module-level cache (loaded once per session) ─────────────────────────────
_loaded_vocab = None


def _download_vocab() -> set:
    """Download 150k+ Urdu word list from urduhack/urdu-words."""
    try:
        import urllib.request
        print("🌐 AenPi: Downloading Urdu vocabulary (150k+ words)... ", end="", flush=True)
        with urllib.request.urlopen(_VOCAB_URL, timeout=15) as response:
            text = response.read().decode("utf-8")
        words = set(line.strip() for line in text.splitlines() if line.strip())
        print(f"✅ {len(words):,} words downloaded.")
        return words
    except Exception as e:
        print(f"\n⚠️  Download failed ({e}). Using built-in vocabulary.")
        return set()


def _save_cache(vocab: set):
    """Save vocabulary to local cache file."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(list(vocab), f, ensure_ascii=False)
    except Exception:
        pass  # cache save failure is non-fatal


def _load_cache() -> set:
    """Load vocabulary from local cache file."""
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            words = set(json.load(f))
        return words
    except Exception:
        return set()


def load_vocabulary(force_download: bool = False) -> set:
    """
    Load Urdu vocabulary with smart caching.

    Priority:
        1. Already loaded this session (memory cache)
        2. Local disk cache (~/.aenpi_cache/urdu_vocab.json)
        3. Download from urduhack/urdu-words (then cache it)
        4. Built-in fallback (~200 words) if all else fails

    Args:
        force_download (bool): Skip cache and re-download. Default False.

    Returns:
        set: Urdu vocabulary words.
    """
    global _loaded_vocab

    # 1. Already in memory
    if _loaded_vocab is not None and not force_download:
        return _loaded_vocab

    # 2. Load from disk cache
    if not force_download and os.path.exists(_CACHE_FILE):
        cached = _load_cache()
        if cached:
            print(f"📂 AenPi: Loaded {len(cached):,} words from cache.")
            _loaded_vocab = cached
            return _loaded_vocab

    # 3. Download and cache
    downloaded = _download_vocab()
    if downloaded:
        _save_cache(downloaded)
        _loaded_vocab = downloaded
        return _loaded_vocab

    # 4. Fallback
    print("📦 AenPi: Using built-in vocabulary.")
    _loaded_vocab = _FALLBACK_VOCABULARY.copy()
    return _loaded_vocab


def clear_cache():
    """Delete the local vocabulary cache to force a fresh download."""
    try:
        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)
            print("🗑️  AenPi: Vocabulary cache cleared.")
        global _loaded_vocab
        _loaded_vocab = None
    except Exception as e:
        print(f"⚠️  Could not clear cache: {e}")


# ─── Edit Distance (Levenshtein) ──────────────────────────────────────────────

def edit_distance(word1: str, word2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Args:
        word1 (str): Source word.
        word2 (str): Target word.

    Returns:
        int: Minimum edits to transform word1 into word2.

    Example:
        >>> edit_distance("احمد", "احمض")
        1
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
    return dp[m][n]


# ─── Spell Corrector ──────────────────────────────────────────────────────────

def spell_correct(word: str, vocabulary: list = None, max_distance: int = 2) -> str:
    """
    Find the closest matching word using edit distance.

    Automatically loads the full 150k Urdu vocabulary if no vocabulary
    is provided. Downloads and caches on first use.

    Args:
        word (str): Potentially misspelled Urdu word.
        vocabulary (list): Optional custom word list. Auto-loads if None.
        max_distance (int): Maximum allowed edit distance. Default 2.

    Returns:
        str: Best matching word, or original word if no match found.

    Example:
        >>> spell_correct("احمض")
        'احمد'
        >>> spell_correct("پاکستاں")
        'پاکستان'
    """
    vocab = vocabulary if vocabulary is not None else list(load_vocabulary())

    best_word = word
    best_dist = max_distance + 1

    for candidate in vocab:
        dist = edit_distance(word, candidate)
        if dist < best_dist:
            best_dist = dist
            best_word = candidate

    return best_word


def spell_correct_text(text: str, vocabulary: list = None, max_distance: int = 2) -> str:
    """
    Apply spell correction to every token in a text.

    Automatically loads full vocabulary if none provided.

    Args:
        text (str): Whitespace-separated Urdu text.
        vocabulary (list): Optional custom word list.
        max_distance (int): Maximum edit distance. Default 2.

    Returns:
        str: Text with each token corrected where possible.

    Example:
        >>> spell_correct_text("احمض گهر میں ہے")
        'احمد گھر میں ہے'
    """
    # Load vocab once for the whole text (not per token)
    vocab = vocabulary if vocabulary is not None else list(load_vocabulary())
    tokens = text.split()
    corrected = [spell_correct(token, vocab, max_distance) for token in tokens]
    return ' '.join(corrected)

"""
AenPi.urdu.spell_corrector
---------------------------
Lightweight Urdu spell correction using Levenshtein edit distance.
Includes a built-in Urdu vocabulary — no external dataset needed.
"""


# ─── Built-in Urdu Vocabulary ─────────────────────────────────────────────────
URDU_VOCABULARY = {
    # Common nouns
    "نام", "گھر", "ملک", "شہر", "گاؤں", "دن", "رات", "وقت", "کام", "بات",
    "جگہ", "راستہ", "دروازہ", "کمرہ", "کتاب", "قلم", "میز", "کرسی", "پانی",
    "کھانا", "روٹی", "چائے", "دودھ", "پھل", "سبزی", "بازار", "دکان", "مکان",
    "درخت", "پھول", "پتہ", "آسمان", "زمین", "سورج", "چاند", "ستارہ", "بادل",
    "ہوا", "بارش", "دریا", "سمندر", "پہاڑ", "جنگل", "باغ", "کھیت", "سڑک",
    "گاڑی", "بس", "ٹرین", "اسکول", "کالج", "یونیورسٹی", "ہسپتال",
    "مسجد", "دفتر", "فیکٹری",

    # People
    "آدمی", "عورت", "لڑکا", "لڑکی", "بچہ", "بچی", "بچے", "ماں", "باپ",
    "بھائی", "بہن", "دادا", "دادی", "نانا", "نانی", "چاچا", "چاچی", "دوست",
    "استاد", "شاگرد", "ڈاکٹر", "انجینئر", "وکیل", "کسان", "مزدور",

    # Common names
    "احمد", "محمد", "علی", "حسن", "عمر", "عثمان", "بلال", "طلحہ", "زید",
    "فاطمہ", "عائشہ", "زینب", "مریم", "ثمینہ", "سارہ", "نور", "حنا",

    # Places
    "پاکستان", "لاہور", "کراچی", "اسلام آباد", "پشاور", "کوئٹہ", "ملتان",
    "فیصل آباد", "راولپنڈی", "حیدرآباد", "ہندوستان", "ایران", "افغانستان",

    # Verbs
    "کرنا", "جانا", "آنا", "کھانا", "پینا", "سونا", "اٹھنا", "بیٹھنا",
    "چلنا", "دوڑنا", "پڑھنا", "لکھنا", "بولنا", "سننا", "دیکھنا", "سمجھنا",
    "جاننا", "مانگنا", "دینا", "لینا", "رکھنا", "ملنا", "بتانا", "سیکھنا",
    "سکھانا", "کھیلنا", "ہنسنا", "رونا", "سوچنا", "پوچھنا",

    # Verb conjugations
    "ہے", "ہیں", "ہوں", "تھا", "تھی", "تھے", "ہوگا", "ہوگی",
    "کیا", "کی", "کیے", "کرتا", "کرتی", "کرتے", "کررہا", "کررہی", "کررہے",
    "گیا", "گئی", "گئے", "آیا", "آئی", "آئے", "رہا", "رہی", "رہے",

    # Adjectives
    "اچھا", "برا", "بڑا", "چھوٹا", "لمبا", "موٹا", "پتلا",
    "خوبصورت", "بدصورت", "مشہور", "نیا", "پرانا", "تیز", "سست", "ہوشیار",
    "امیر", "غریب", "خوش", "ناخوش", "صحیح", "غلط", "آسان", "مشکل",
    "گرم", "ٹھنڈا", "میٹھا", "کڑوا", "سخت", "نرم", "صاف", "گندا", "روشن",

    # Numbers
    "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو", "دس",
    "بیس", "تیس", "چالیس", "پچاس", "سو", "ہزار", "لاکھ", "کروڑ",

    # Time
    "آج", "کل", "پرسوں", "صبح", "دوپہر", "شام", "رات", "ابھی", "پھر",
    "پہلے", "بعد", "جلدی", "دیر", "ہمیشہ", "کبھی",

    # Common words
    "بہت", "کم", "زیادہ", "تھوڑا", "سب", "کچھ", "سارا", "پورا", "ساتھ",
    "بغیر", "لیے", "تک", "سے", "میں", "پر", "کے", "کی", "کو", "نے",
    "بھی", "تو", "ہی", "نہیں", "نہ", "ہاں", "جی", "شکریہ", "معاف",
    "اللہ", "انسان", "دنیا", "زندگی", "موت", "محبت", "نفرت",
    "خوشی", "غم", "امید", "خوف", "سچ", "جھوٹ", "عزت", "محنت", "قسمت",
}


# ─── Edit Distance (Levenshtein) ──────────────────────────────────────────────

def edit_distance(word1: str, word2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

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
    Find the closest word in a vocabulary using edit distance.
    Uses built-in Urdu vocabulary if none is provided.

    Args:
        word (str): Potentially misspelled Urdu word.
        vocabulary (list): Optional custom word list. Uses URDU_VOCABULARY if None.
        max_distance (int): Maximum allowed edit distance. Default 2.

    Returns:
        str: Best matching word, or original word if no match found.

    Example:
        >>> spell_correct("احمض")
        'احمد'
        >>> spell_correct("پاکستاں")
        'پاکستان'
    """
    vocab = vocabulary if vocabulary is not None else list(URDU_VOCABULARY)

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
    Uses built-in Urdu vocabulary if none is provided.

    Args:
        text (str): Whitespace-separated Urdu text.
        vocabulary (list): Optional custom word list.
        max_distance (int): Maximum edit distance for correction.

    Returns:
        str: Text with each token corrected if possible.

    Example:
        >>> spell_correct_text("احمض گهر میں ہے")
        'احمد گھر میں ہے'
    """
    tokens = text.split()
    corrected = [spell_correct(token, vocabulary, max_distance) for token in tokens]
    return ' '.join(corrected)

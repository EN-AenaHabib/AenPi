"""
AenPi.urdu.spell_corrector
---------------------------
Lightweight Urdu spell correction using Levenshtein edit distance.
No external libraries required — pure Python implementation.
"""


# ─── Edit Distance (Levenshtein) ──────────────────────────────────────────────

def edit_distance(word1: str, word2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Operations: insert, delete, substitute (each costs 1).

    Args:
        word1 (str): Source word.
        word2 (str): Target word.

    Returns:
        int: Minimum edits to transform word1 → word2.

    Example:
        >>> edit_distance("احمد", "احمض")
        1
    """
    m, n = len(word1), len(word2)

    # dp[i][j] = edit distance between word1[:i] and word2[:j]
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
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1],  # substitute
                )

    return dp[m][n]


# ─── Spell Corrector ──────────────────────────────────────────────────────────

def spell_correct(word: str, vocabulary: list, max_distance: int = 2) -> str:
    """
    Find the closest word in a vocabulary using edit distance.

    Args:
        word (str): Potentially misspelled Urdu word.
        vocabulary (list): List of known correct Urdu words.
        max_distance (int): Maximum allowed edit distance. Default 2.

    Returns:
        str: Best matching word from vocabulary, or original word if
             no candidate within max_distance is found.

    Example:
        >>> spell_correct("احمض", ["احمد", "محمد", "علی"])
        'احمد'
    """
    if not vocabulary:
        return word

    best_word = word
    best_dist = max_distance + 1  # start above threshold

    for candidate in vocabulary:
        dist = edit_distance(word, candidate)
        if dist < best_dist:
            best_dist = dist
            best_word = candidate

    return best_word


def spell_correct_text(text: str, vocabulary: list, max_distance: int = 2) -> str:
    """
    Apply spell correction to every token in a text.

    Args:
        text (str): Whitespace-separated Urdu text.
        vocabulary (list): List of known correct words.
        max_distance (int): Maximum edit distance for correction.

    Returns:
        str: Text with each token corrected if possible.

    Example:
        >>> vocab = ["احمد", "محمد", "علی", "گھر", "ہے"]
        >>> spell_correct_text("احمض گهر ہے", vocab)
        'احمد گھر ہے'
    """
    tokens = text.split()
    corrected = [spell_correct(token, vocabulary, max_distance) for token in tokens]
    return ' '.join(corrected)

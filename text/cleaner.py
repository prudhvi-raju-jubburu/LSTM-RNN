"""
Text cleaning and normalization pipeline for TTS.
"""
import re

# Common English abbreviations mapping
ABBREVIATIONS = [
    (r"\bMr\.", "Mister"),
    (r"\bMrs\.", "Missus"),
    (r"\bMs\.", "Miss"),
    (r"\bDr\.", "Doctor"),
    (r"\bProf\.", "Professor"),
    (r"\bSr\.", "Senior"),
    (r"\bJr\.", "Junior"),
    (r"\bSt\.", "Saint"),
    (r"\bCo\.", "Company"),
    (r"\bCorp\.", "Corporation"),
    (r"\bInc\.", "Incorporated"),
    (r"\bLtd\.", "Limited"),
    (r"\bvs\.", "versus"),
    (r"\betc\.", "etcetera"),
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\bapprox\.", "approximately"),
    (r"\bdept\.", "department"),
    (r"\bft\.", "feet"),
    (r"\bkg\b", "kilograms"),
    (r"\bkm\b", "kilometers"),
    (r"\bhr\b", "hour"),
    (r"\bmin\b", "minute"),
    (r"\bsec\b", "second"),
]

# Simple number to word converter for integers 0-9999
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """Convert an integer (0 to 999,999) to written English words."""
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + number_to_words(-n)

    parts = []
    if n >= 1000:
        thousands = n // 1000
        parts.append(number_to_words(thousands) + " thousand")
        n %= 1000

    if n >= 100:
        hundreds = n // 100
        parts.append(ONES[hundreds] + " hundred")
        n %= 100

    if n >= 20:
        tens = n // 10
        rem = n % 10
        if rem > 0:
            parts.append(TENS[tens] + " " + ONES[rem])
        else:
            parts.append(TENS[tens])
    elif n >= 10:
        parts.append(TEENS[n - 10])
    elif n > 0:
        parts.append(ONES[n])

    return " ".join(parts).strip()


def expand_numbers(text: str) -> str:
    """Replace numeric sequences with English word representations."""
    def _replace(match):
        val_str = match.group(0)
        try:
            val = int(val_str)
            if val < 1000000:
                return number_to_words(val)
            return " ".join(ONES[int(digit)] if int(digit) != 0 else "zero" for digit in val_str)
        except ValueError:
            return val_str

    return re.sub(r"\b\d+\b", _replace, text)


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations into full spoken forms."""
    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """
    Complete text cleaning pipeline:
    1. Expand abbreviations
    2. Normalize whitespace and newlines
    3. Expand numbers to spoken words
    4. Convert to lowercase
    5. Filter to allowable symbols
    """
    if not text:
        return ""

    # Collapse newlines and tabs
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Expand abbreviations
    text = expand_abbreviations(text)

    # Expand numbers
    text = expand_numbers(text)

    # Convert to lowercase
    text = text.lower()

    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text).strip()

    return text

"""
Defines the vocabulary and symbols used for character-level TTS tokenization.
"""

# Special tokens
PAD = "_"
EOS = "~"
UNK = "^"

# Punctuations and alphanumeric characters
PUNCTUATION = "!'(),-.:;? "
LETTERS = "abcdefghijklmnopqrstuvwxyz"
NUMBERS = "0123456789"

# Complete symbol list
SYMBOLS = [PAD, EOS, UNK] + list(PUNCTUATION) + list(LETTERS) + list(NUMBERS)

# Bidirectional mapping
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

PAD_ID = SYMBOL_TO_ID[PAD]
EOS_ID = SYMBOL_TO_ID[EOS]
UNK_ID = SYMBOL_TO_ID[UNK]
NUM_SYMBOLS = len(SYMBOLS)

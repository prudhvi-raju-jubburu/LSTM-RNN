"""
Tokenizer mapping text strings to sequence tensors and back.
"""
from typing import List
import torch
from text.symbols import SYMBOL_TO_ID, ID_TO_SYMBOL, UNK_ID, EOS_ID, EOS
from text.cleaner import clean_text


def text_to_sequence(text: str, add_eos: bool = True) -> List[int]:
    """
    Clean text and convert characters to token IDs.
    """
    cleaned = clean_text(text)
    sequence = []

    for char in cleaned:
        seq_id = SYMBOL_TO_ID.get(char, UNK_ID)
        sequence.append(seq_id)

    if add_eos:
        sequence.append(EOS_ID)

    return sequence


def sequence_to_text(sequence: List[int]) -> str:
    """
    Convert a list of token IDs back into a string.
    """
    chars = []
    for token_id in sequence:
        if token_id in ID_TO_SYMBOL:
            symbol = ID_TO_SYMBOL[token_id]
            if symbol == EOS:
                break
            chars.append(symbol)
    return "".join(chars)


def text_to_tensor(text: str, add_eos: bool = True, device: str = "cpu") -> torch.Tensor:
    """
    Convenience function returning a 1D or 2D PyTorch LongTensor.
    """
    seq = text_to_sequence(text, add_eos=add_eos)
    return torch.tensor(seq, dtype=torch.long, device=device)

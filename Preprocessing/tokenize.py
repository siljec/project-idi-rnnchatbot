import re

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\":;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens"""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Returns:
    a list of integers, the token-ids for the sentence.
    """
    words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]

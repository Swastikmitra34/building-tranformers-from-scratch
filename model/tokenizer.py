class CharTokenizer:
    """
    Very simple character-level tokenizer.

    - Builds vocabulary from given text.
    - Provides encode (string -> list[int])
      and decode (list[int] -> string).
    """
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)

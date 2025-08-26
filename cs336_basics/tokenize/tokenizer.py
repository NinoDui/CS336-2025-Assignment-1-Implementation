from collections.abc import Iterable
import itertools

import regex as re

from cs336_basics.common import constants as C, io, types as T
from cs336_basics.tokenize import bpe, pretoken as pre


class Tokenizer:
    def __init__(self, vocab: T.VocabType, merges: T.MergeType, *, special_tokens: list[str] | None = None, **kwargs):
        super().__init__()

        self._vocab: T.VocabType = vocab
        self._merges: T.MergeType = merges
        self._special_tokens = special_tokens

        self._reverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}

        if special_tokens is not None:
            for special_token in special_tokens:
                if (token_bytes := special_token.encode("utf-8")) not in self._reverse_vocab:
                    idx = len(self._vocab)
                    self._vocab[idx] = token_bytes
                    self._reverse_vocab[token_bytes] = idx

    @classmethod
    def from_file(
        cls, vocab_filepath: T.PathLike, merges_filepath: T.PathLike, *, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        vocab_content = io.load_vocab(vocab_filepath)
        vocab = {int(k): v for k, v in vocab_content.items()}

        merges_content = io.load_text(merges_filepath)
        merges = []
        for line in merges_content.split("\n"):
            if not line or len(line.strip()) == 0:
                break
            p1, p2, *_ = line.split()
            merges.append((p1.encode("utf-8"), p2.encode("utf-8")))
        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        token_sections, _ = pre.pretoken(text, special_tokens=self._special_tokens, split_pattern=re.compile(C.PAT))

        result = []
        for seg in token_sections:
            result.extend(self._encode_tokens(seg))
        return result

    def _encode_tokens(self, tokens: list[T.BytesToken]) -> list[int]:
        return list(itertools.chain.from_iterable(self._token_to_ids(token) for token in tokens))

    def _token_to_ids(self, token: T.BytesToken, *, requie_merge: bool = True) -> list[int]:
        if requie_merge:
            token = self._apply_merges_to_token(token)
        return [self._reverse_vocab[b] for b in token]

    def _apply_merges_to_token(self, token: T.BytesToken) -> T.BytesToken:
        for merge in self._merges:
            if len(token) == 1:
                # Current token is fully merged, only 1 left in tuple
                break
            token = bpe._merge_pair_in_token(token, merge)
        return token

    def encode_iterable(self, texts: Iterable[str]) -> Iterable[int]:
        it = iter(texts)
        while True:
            try:
                text = next(it)
            except StopIteration:
                break
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return b"".join(self._vocab[id] for id in token_ids).decode("utf-8", errors="replace")

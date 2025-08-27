import logging

import regex as re

from cs336_basics.common import constants as C, types as T
from cs336_basics.tokenize import pretoken as pre
from cs336_basics.tokenize.heap import MaxHeap

logger = logging.getLogger(__name__)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], **kwargs) -> tuple[T.VocabType, T.MergeType]:
    vocab = {i: bytes([i]) for i in range(C.DEFAULT_BYTE_SIZE)}
    for delta, special_token in enumerate(special_tokens):
        vocab[C.DEFAULT_BYTE_SIZE + delta] = special_token.encode("utf-8")
    logger.info(f"Initialized vocab with {len(vocab)} tokens")
    merges = []

    # Calculate the frequency of each token
    num_processes = kwargs.get("num_processes", 1)
    if num_processes > 1:
        token_freq = pre.pretoken_and_count_in_parallel(
            input_path, special_tokens=special_tokens, split_pattern=re.compile(C.PAT), num_processes=num_processes
        )
    else:
        # TODO: align the API behavior with parallel mode
        with open(input_path, "rb") as f:
            token_freq = pre.pretoken_and_count(
                f.read(), special_tokens=special_tokens, split_pattern=re.compile(C.PAT)
            )

    pair_to_cnt, pair_to_token = _establish_pair_cache(token_freq)

    while len(vocab) < vocab_size:
        heap = MaxHeap(pair_to_cnt, key=lambda x: (x[-1], x[0]))
        if heap.is_empty():
            logger.info(f"No more pairs to merge, stopping BPE training with {len(vocab)} tokens")
            break
        top_freq_pair: T.BytesPair = heap.pop()[0]  # type: ignore[assignment]

        new_token_bytes = b"".join(top_freq_pair)
        vocab[len(vocab)] = new_token_bytes
        merges.append(top_freq_pair)

        affected_tokens = pair_to_token[top_freq_pair].copy()
        for token in affected_tokens:
            new_token = _merge_pair_in_token(token, top_freq_pair, repr=new_token_bytes)
            token_freq[new_token] = token_freq[token]

            for old_pair in zip(token[:-1], token[1:], strict=False):
                pair_to_cnt[old_pair] -= token_freq[token]
                if token in pair_to_token[old_pair]:
                    # the token here is that affected by the top_freq_pair
                    # thus, not all the pairs in this token are affected
                    # which means we should check whether it is "affected" or inside the pair_to_token
                    pair_to_token[old_pair].remove(token)

            for new_pair in zip(new_token[:-1], new_token[1:], strict=False):
                pair_to_cnt[new_pair] = pair_to_cnt.get(new_pair, 0) + token_freq[new_token]
                pair_to_token.setdefault(new_pair, set()).add(new_token)

        pair_to_cnt.pop(top_freq_pair)
        pair_to_token.pop(top_freq_pair)

    return vocab, merges


def _establish_pair_cache(
    token_freq: dict[T.BytesToken, int],
) -> tuple[dict[T.BytesPair, int], dict[T.BytesPair, set[T.BytesToken]]]:
    """Establish the pair cache,
    1. Pair to Count, e.g., ('lo', 's') -> 5 in bytes
    2. Pair to Pretokens containing the pair, e.g., ('l', 'o') -> {'los', 'low'} in bytes

    Args:
        token_to_cnt (dict[T.BytesToken, int]): The frequency of each token

    Returns:
        tuple[dict[T.BytesPair, int], dict[T.BytesPair, set[T.BytesToken]]]:
            {pair: count}, {pair: {tokens containing the pair}}
    """
    pair_freq: dict[T.BytesPair, int] = {}
    pair_to_token: dict[T.BytesPair, set[T.BytesToken]] = {}
    for token, cnt in token_freq.items():
        for pair in zip(token[:-1], token[1:], strict=False):
            pair_freq[pair] = pair_freq.get(pair, 0) + cnt
            pair_to_token.setdefault(pair, set()).add(token)
    return pair_freq, pair_to_token


def _merge_pair_in_token(token: T.BytesToken, pair: T.BytesPair, repr: bytes | None = None) -> T.BytesToken:
    """Merge the pair by replacing the pair with the new token in the word token

    Args:
        token (T.BytesToken): The word token to merge the pair in
        pair (T.BytesPair): The pair to merge

    Returns:
        T.BytesToken: The new word token with the pair merged
    """
    repr = repr or b"".join(pair)
    elems, idx = [], 0
    while idx < len(token):
        if idx < len(token) - 1 and token[idx] == pair[0] and token[idx + 1] == pair[1]:
            elems.append(repr)
            idx += 2
        else:
            elems.append(token[idx])
            idx += 1
    return tuple(elems)

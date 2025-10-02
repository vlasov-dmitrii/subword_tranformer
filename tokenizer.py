import tiktoken
import regex as re

class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.vocab_size = 0

    def get_stats(self, ids):
        freq = {}
        for pair in zip(ids, ids[1:]):
            freq[pair] = freq.get(pair, 0) + 1
        return freq

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size, verbose=False):
        tokens = list(text.encode("utf-8"))
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.vocab_size = vocab_size

        next_id = 256
        while next_id < vocab_size:
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            self.merges[pair] = next_id
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            tokens = self.merge(tokens, pair, next_id)

            if verbose:
                print(f"merge {pair} -> {next_id}, token: {self.vocab[next_id]}")

            next_id += 1

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while True:
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|
                          [^\r\n\p{L}\p{N}]?+\p{L}+|
                          \p{N}{1,3}|
                          [^\s\p{L}\p{N}]++[\r\n]*|
                          \s*[\r\n]|
                          \s+(?!\S)|
                          \s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = re.compile(pattern, re.VERBOSE)

    def _split(self, text):
        return [m.group(0) for m in self.pattern.finditer(text)]

    def train(self, text, vocab_size, verbose=False):
        parts = self._split(text)

        tokens = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.vocab_size = vocab_size
        next_id = 256

        for part in parts:
            part_tokens = list(part.encode("utf-8"))
            while next_id < vocab_size:
                stats = self.get_stats(part_tokens)
                if not stats:
                    break
                pair = max(stats, key=stats.get)
                if pair in self.merges:
                    part_tokens = self.merge(part_tokens, pair, self.merges[pair])
                    continue
                self.merges[pair] = next_id
                self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
                part_tokens = self.merge(part_tokens, pair, next_id)
                if verbose:
                    print(f"merge {pair} -> {next_id}, token: {self.vocab[next_id]}")
                next_id += 1
            tokens.extend(part_tokens)

    def encode(self, text):
        parts = self._split(text)
        out = []
        for part in parts:
            tokens = list(part.encode("utf-8"))
            while True:
                stats = self.get_stats(tokens)
                if not stats:
                    break
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                tokens = self.merge(tokens, pair, idx)
            out.extend(tokens)
        return out

class GPT4Tokenizer:
    def __init__(self, encoding_name="cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)

        self.mergeable_ranks = self.enc._mergeable_ranks
        self.merges = self.recover_merges(self.mergeable_ranks)

        self.byte_shuffle = {i: self.mergeable_ranks[bytes([i])] for i in range(256)}
        self.inv_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

        self.vocab = {rank: token for token, rank in self.mergeable_ranks.items()}

    def recover_merges(self, mergeable_ranks):
        merges = {}
        def bpe_split(token, max_rank):
            for i in range(1, len(token)):
                left = token[:i]
                right = token[i:]
                if left in mergeable_ranks and right in mergeable_ranks:
                    if mergeable_ranks[left] <= max_rank and mergeable_ranks[right] <= max_rank:
                        return left, right
            return None

        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                continue
            pair = bpe_split(token, rank)
            if pair:
                left, right = pair
                ix0 = mergeable_ranks[left]
                ix1 = mergeable_ranks[right]
                merges[(ix0, ix1)] = rank
        return merges

    def encode(self, text):
        bts = text.encode("utf-8")
        tokens = [self.byte_shuffle[b] for b in bts]

        while True:
            stats = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    stats[pair] = self.merges[pair]
            if not stats:
                break
            pair_to_merge = min(stats, key=stats.get)
            idx = stats[pair_to_merge]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        def expand(idx):
            token = self.vocab[idx]
            if len(token) == 1:
                return [self.inv_byte_shuffle[idx]]
            for pair, rank in self.merges.items():
                if rank == idx:
                    left, right = pair
                    return expand(left) + expand(right)
            return list(token)

        bts = []
        for idx in ids:
            bts.extend(expand(idx))
        return bytes(bts).decode("utf-8", errors="replace")
    


class GPT4TokenizerWithSpecial(GPT4Tokenizer):
    def __init__(self, encoding_name="cl100k_base", allowed_special=None):
        super().__init__(encoding_name)
        self.allowed_special = allowed_special
        self.special_id_to_token = {}
        if allowed_special == "all":
            for tok, idx in self.enc._special_tokens.items():
                self.special_id_to_token[idx] = tok

    def _expand_id_to_bytes(self, idx):
        token = self.vocab[idx]
        if len(token) == 1:
            return [self.inv_byte_shuffle[idx]]
        for pair, rank in self.merges.items():
            if rank == idx:
                left, right = pair
                return self._expand_id_to_bytes(left) + self._expand_id_to_bytes(right)
        return list(token)

    def encode(self, text):
        import re
        if self.allowed_special == "all":
            special_pattern = "|".join(re.escape(tok) for tok in self.special_id_to_token.values())
            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]

        out_ids = []
        for part in parts:
            if part in self.special_id_to_token.values():
                out_ids.append({v:k for k,v in self.special_id_to_token.items()}[part])
            else:
                out_ids.extend(super().encode(part))
        return out_ids

    def decode(self, ids):
        bts = []
        for idx in ids:
            if idx in self.special_id_to_token:
                bts.extend(self.special_id_to_token[idx].encode("utf-8"))
            else:
                bts.extend(self._expand_id_to_bytes(idx))
        return bytes(bts).decode("utf-8", errors="replace")
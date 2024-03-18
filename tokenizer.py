import pickle
from collections import Counter
import regex as re


class BPETokenizer:
    def __init__(self, pattern: str, vocab_size: int):
        assert vocab_size >= 256

        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.pattern = pattern

    def train(self, text: str):
        max_iters = self.vocab_size - 256
        split_text = re.findall(self.pattern, text)
        encoded_chunks = [(text_chunk.encode("utf-8")) for text_chunk in split_text]
        chunk_freqs = Counter(encoded_chunks)
        chunk_splits = {chunk: list(chunk) for chunk in encoded_chunks}

        num_iters = 0
        while num_iters < max_iters:
            # get stats
            pair_freqs = {}
            for chunk, splits in chunk_splits.items():
                # updates pair_freqs in place
                self.get_stats(splits, pair_freqs, chunk_freqs[chunk])

            # merge
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_idx = 256 + num_iters
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = 256 + num_iters

            new_chunk_splits, new_chunk_freqs = {}, {}
            for chunk, splits in chunk_splits.items():
                new_splits = self.merge(splits, best_pair, new_idx)
                new_chunk = b"".join(self.vocab[id] for id in new_splits)
                new_chunk_splits[new_chunk] = new_splits
                new_chunk_freqs[new_chunk] = chunk_freqs[chunk]

            chunk_splits, chunk_freqs = new_chunk_splits, new_chunk_freqs
            num_iters += 1

    def get_stats(
        self, ids: list[int], stats: dict[tuple[int, int], int] = {}, freq: int = 1
    ) -> dict[tuple[int, int], int]:
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + freq

        return stats

    def merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        updated_ids = []

        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                # Merge the pair
                updated_ids.append(new_id)
                i += 2
            else:
                # If not merged, keep the original subword ID
                updated_ids.append(ids[i])
                i += 1

        return updated_ids

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "merges": self.merges}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            tok_data = pickle.load(f)
            self.vocab = tok_data["vocab"]
            self.merges = tok_data["merges"]

    def encode(self, text: str) -> list[int]:
        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)

        return ids

    def _encode_chunk(self, text_chunk: str) -> list[int]:
        tokens = list(text_chunk.encode("utf-8"))
        while len(tokens) >= 2:
            pairs = self.get_stats(tokens, {})

            potential_merge = min(
                pairs, key=lambda pair: self.merges.get(pair, float("inf"))
            ) 
            if potential_merge not in self.merges:
                break

            # Merge the pair in our text
            tokens = self.merge(tokens, potential_merge, self.merges[potential_merge])

        return tokens

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join([self.vocab[idx] for idx in ids])
        return tokens.decode("utf-8", errors="replace")

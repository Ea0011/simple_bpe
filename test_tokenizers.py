from tokenizer import BPETokenizer
import regex as re

sample_text = "I am"
pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def test_encode():
    tokenizer = BPETokenizer(pattern=pattern, vocab_size=256)
    tokenizer.encode(sample_text)

def test_decode():
    tokenizer = BPETokenizer(pattern=pattern, vocab_size=256)
    tokenizer.decode([0, 1, 2])

def test_encode_decode():
    tokenizer = BPETokenizer(pattern=pattern, vocab_size=256)
    assert tokenizer.decode(tokenizer.encode(sample_text)) == sample_text

def test_initial_vocab():
    assert BPETokenizer(pattern=pattern, vocab_size=256).vocab == { idx: bytes([idx]) for idx in range(256) }

def test_train():
    text_to_train = "IIII"
    tokenizer = BPETokenizer(pattern=pattern, vocab_size=257)
    tokenizer.train(text=text_to_train)

    assert tokenizer.merges == {(int.from_bytes(b"I"), int.from_bytes(b"I")): 256}
    assert 256 in [idx for idx, _ in tokenizer.vocab.items()]
    assert (b"I" + b"I") in [byte for _, byte in tokenizer.vocab.items()]
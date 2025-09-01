import pytest

from cs336_basics.common import io


@pytest.fixture
def file_path():
    return "tests/fixtures/paragraph.txt"


@pytest.mark.parametrize(
    "separator, chunk_size, expected",
    [
        ([b" "], 512, 508),
        ([b"\n"], 512, 495),
        ([b"\n", b" "], 512, 495),
        ([b" ", b"<|endoftext|>"], 512, 508),
        ([b"<|endoftext|>"], 1024, 882),
        ([b"<startoftext>"], 1024, 1024),
    ],
)
def test_read_until_in_bytes(file_path, separator, chunk_size, expected):
    iter = io.read_until(file_path, separator, chunk_size=chunk_size, bytes_mode=True)
    content = next(iter)

    assert len(content) == expected


@pytest.mark.parametrize(
    "separator, chunk_size, expected",
    [
        ([" "], 512, 508),
        (["\n"], 512, 495),
        (["\n", " "], 512, 495),
        ([" ", "<|endoftext|>"], 1024, 882),
        (["<startoftext>"], 512, 512),
    ],
)
def test_read_until_in_str(file_path, separator, chunk_size, expected):
    iter = io.read_until(file_path, separator, chunk_size=chunk_size, bytes_mode=False)
    content = next(iter)
    assert len(content) == expected

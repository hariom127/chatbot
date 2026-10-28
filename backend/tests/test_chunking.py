import pytest

from app.services.chunking import chunk_document, clean_text, split_text


def make_doc(text):
    return {"id": "d1", "title": "T", "url": "http://x", "modified_time": "2024-01-01T00:00:00Z", "text": text}


def test_clean_text():
    assert clean_text("a\r\n\r\n\r\n\r\nb  c\t\td ") == "a\n\nb c d"


def test_empty_document_gives_no_chunks():
    assert chunk_document(make_doc("   \n\n "), 100, 10) == []


def test_short_document_single_chunk():
    chunks = chunk_document(make_doc("hello world"), 100, 10)
    assert len(chunks) == 1 and chunks[0].text == "hello world"


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    pieces = split_text(text, 200, 40)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    # overlap: the start of the next chunk repeats the tail of the previous one
    assert pieces[1].split()[0] in pieces[0]
    assert " ".join(pieces).count("word299") >= 1


def test_metadata_and_stable_ids():
    chunks = chunk_document(make_doc("x " * 500), 200, 20)
    assert [c.id for c in chunks] == [f"d1:{i}" for i in range(len(chunks))]
    for i, c in enumerate(chunks):
        assert c.metadata == {
            "doc_id": "d1", "title": "T", "url": "http://x",
            "chunk_index": i, "modified_time": "2024-01-01T00:00:00Z",
        }


def test_invalid_params():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)

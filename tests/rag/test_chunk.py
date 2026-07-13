"""Chunker: heading awareness, sliding windows, citation offsets."""

from __future__ import annotations

from agentloop.rag.chunk import chunk_text


class TestMarkdownAware:
    def test_sections_split_on_headings(self):
        content = "# One\n\nalpha\n\n## Two\n\nbeta\n\n# Three\n\ngamma\n"
        chunks = chunk_text(content, size=512, overlap=0)
        assert [c.heading for c in chunks] == ["One", "Two", "Three"]
        assert "alpha" in chunks[0].text
        assert "beta" in chunks[1].text

    def test_preamble_before_first_heading_kept(self):
        content = "intro text\n\n# One\n\nbody\n"
        chunks = chunk_text(content)
        assert chunks[0].heading == ""
        assert "intro" in chunks[0].text

    def test_plain_text_is_one_chunk(self):
        chunks = chunk_text("no headings here at all")
        assert len(chunks) == 1
        assert chunks[0].heading == ""

    def test_empty_content_no_chunks(self):
        assert chunk_text("") == []


class TestWindows:
    def test_oversized_section_gets_sliding_windows_with_overlap(self):
        body = "word " * 2000  # 10k chars
        content = f"# Big\n\n{body}"
        chunks = chunk_text(content, size=256, overlap=64)  # 1024/768 chars
        assert len(chunks) > 5
        assert all(c.heading == "Big" for c in chunks)
        # consecutive windows overlap by ~overlap*4 chars
        assert chunks[1].start < chunks[0].end

    def test_offsets_reproduce_text_exactly(self):
        content = "# A\n\n" + "alpha bravo " * 500 + "\n\n# B\n\ncharlie"
        for chunk in chunk_text(content, size=128, overlap=16):
            assert content[chunk.start : chunk.end] == chunk.text  # citation-grade

    def test_indices_are_sequential(self):
        content = "# A\n\n" + "x " * 3000
        chunks = chunk_text(content, size=128, overlap=16)
        assert [c.index for c in chunks] == list(range(len(chunks)))

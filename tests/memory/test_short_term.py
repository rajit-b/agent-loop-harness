"""Short-term tier: episodic log, overflow summarization, history assembly."""

from __future__ import annotations

from .conftest import make_manager


class TestHistoryAssembly:
    async def test_history_is_summary_then_verbatim_tail(self, mem_store, embedder):
        manager, _ = make_manager(mem_store, embedder, [])
        mem_store.set_summary("session-1", "We discussed the loader.")
        mem_store.append_episodic("session-1", "t1", "user", "and budgets?")
        mem_store.append_episodic("session-1", "t1", "assistant", "checked at boundaries")
        messages = manager.history_messages()
        assert len(messages) == 3
        assert "Summary of this session" in messages[0].text
        assert "We discussed the loader." in messages[0].text
        assert messages[1].text == "and budgets?" and messages[1].role == "user"
        assert messages[2].role == "assistant"

    async def test_sessions_are_isolated(self, mem_store, embedder):
        manager, _ = make_manager(mem_store, embedder, [], session_id="session-2")
        mem_store.append_episodic("session-1", "t", "user", "other session content")
        assert manager.history_messages() == []


class TestOverflowSummarization:
    async def test_under_budget_no_model_call(self, mem_store, embedder):
        manager, provider = make_manager(mem_store, embedder, [], max_tokens=1000)
        mem_store.append_episodic("session-1", "t", "user", "short line")
        assert await manager.maybe_summarize() is False
        assert provider.requests == []

    async def test_overflow_folds_oldest_half_into_summary(self, mem_store, embedder):
        # max 50 tokens ≈ 200 chars; six 100-char rows = ~150 tokens
        manager, provider = make_manager(
            mem_store, embedder, ["condensed summary of old lines"], max_tokens=50
        )
        for i in range(6):
            mem_store.append_episodic(
                "session-1", f"t{i}", "user", f"line-{i} " + "x" * 90
            )
        assert await manager.maybe_summarize() is True

        # the summarizer saw the OLDEST lines
        request = provider.requests[0]
        assert "line-0" in request.messages[1].text
        assert "line-5" not in request.messages[1].text

        # summary stored; folded rows excluded; verbatim tail kept
        assert mem_store.get_summary("session-1") == "condensed summary of old lines"
        remaining = mem_store.episodic_rows("session-1")
        assert [r.content.split()[0] for r in remaining] == [
            "line-3", "line-4", "line-5"
        ]
        messages = manager.history_messages()
        assert "condensed summary" in messages[0].text
        assert messages[1].text.startswith("line-3")

    async def test_resummarize_merges_previous_summary(self, mem_store, embedder):
        manager, provider = make_manager(
            mem_store, embedder, ["updated summary"], max_tokens=50
        )
        mem_store.set_summary("session-1", "earlier summary")
        for i in range(4):
            mem_store.append_episodic("session-1", "t", "user", "y" * 100)
        await manager.maybe_summarize()
        assert "earlier summary" in provider.requests[0].messages[1].text

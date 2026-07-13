"""Consolidation: extraction, promotion rules, dedup-merge, provenance."""

from __future__ import annotations

from .conftest import candidate, extraction, make_manager


def seed_turn(mem_store, session: str, text: str) -> None:
    mem_store.append_episodic(session, "t1", "user", text)


class TestExplicitPromotion:
    async def test_explicit_directive_promotes_on_first_sight(
        self, mem_store, embedder, emitter
    ):
        manager, _ = make_manager(
            mem_store, embedder,
            [extraction(candidate(
                "User prefers tabs over spaces", explicit=True,
                quote="always use tabs",
            ))],
            emitter=emitter,
        )
        seed_turn(mem_store, "session-1", "always use tabs in this repo")
        counts = await manager.consolidate("t1")
        assert counts == {"extracted": 1, "merged": 0, "added": 1, "promoted": 1}
        [fact] = mem_store.active_facts()
        assert fact.status == "active"
        assert fact.text == "User prefers tabs over spaces"
        # provenance rides along (§11)
        assert fact.source_session == "session-1"
        assert fact.source_quote == "always use tabs"
        assert any(k == "memory.fact_promoted" for k, _ in emitter.events)

    async def test_directive_regex_is_a_safety_net(self, mem_store, embedder):
        """Extractor said explicit=False but the quote is a directive."""
        manager, _ = make_manager(
            mem_store, embedder,
            [extraction(candidate(
                "User prefers rebase workflows", explicit=False,
                quote="I prefer rebase over merge",
            ))],
        )
        seed_turn(mem_store, "session-1", "I prefer rebase over merge")
        await manager.consolidate("t1")
        assert len(mem_store.active_facts()) == 1


class TestRecurrencePromotion:
    async def test_single_nonexplicit_mention_stays_candidate(
        self, mem_store, embedder
    ):
        manager, _ = make_manager(
            mem_store, embedder,
            [extraction(candidate("Project deploys on config fridays", confidence=0.95))],
        )
        seed_turn(mem_store, "session-1", "we deployed friday again")
        counts = await manager.consolidate("t1")
        assert counts["added"] == 1 and counts["promoted"] == 0
        assert mem_store.active_facts() == []
        assert mem_store.get_fact(1).status == "candidate"

    async def test_second_session_promotes(self, mem_store, embedder):
        fact = candidate("Project deploys on config fridays", confidence=0.95)
        manager1, _ = make_manager(
            mem_store, embedder, [extraction(fact)], session_id="session-1"
        )
        seed_turn(mem_store, "session-1", "deployed friday")
        await manager1.consolidate("t1")

        manager2, _ = make_manager(
            mem_store, embedder, [extraction(fact)], session_id="session-2"
        )
        seed_turn(mem_store, "session-2", "deployed friday again")
        counts = await manager2.consolidate("t1")
        assert counts["merged"] == 1 and counts["promoted"] == 1
        [promoted] = mem_store.active_facts()
        assert promoted.session_count == 2
        assert mem_store.fact_count() == 1  # merged, not duplicated

    async def test_third_mention_in_one_session_promotes(self, mem_store, embedder):
        fact = candidate("Project deploys on config fridays", confidence=0.95)
        manager, _ = make_manager(
            mem_store, embedder,
            [extraction(fact), extraction(fact), extraction(fact)],
        )
        for i in range(3):
            seed_turn(mem_store, "session-1", f"mention {i}")
            await manager.consolidate(f"t{i}")
        [promoted] = mem_store.active_facts()
        assert promoted.support_count == 3
        assert promoted.session_count == 1  # promoted via mentions, not sessions

    async def test_low_confidence_never_promotes_by_recurrence(
        self, mem_store, embedder
    ):
        fact = candidate("Maybe the config loader is slow", confidence=0.3)
        manager, _ = make_manager(
            mem_store, embedder,
            [extraction(fact)] * 4,
        )
        for i in range(4):
            seed_turn(mem_store, "session-1", f"mention {i}")
            await manager.consolidate(f"t{i}")
        assert mem_store.active_facts() == []


class TestDedup:
    async def test_distinct_facts_insert_separately(self, mem_store, embedder):
        manager, _ = make_manager(
            mem_store, embedder,
            [extraction(
                candidate("User prefers tabs", quote="always tabs"),
                candidate("Jira board tracks the provider work",
                          type="fact", quote=""),
            )],
        )
        seed_turn(mem_store, "session-1", "misc")
        counts = await manager.consolidate("t1")
        assert counts["added"] == 2 and counts["merged"] == 0

    async def test_cursor_prevents_reprocessing(self, mem_store, embedder):
        manager, provider = make_manager(
            mem_store, embedder, [extraction(candidate("x config y"))]
        )
        seed_turn(mem_store, "session-1", "line")
        await manager.consolidate("t1")
        # nothing new appended → no extractor call at all
        counts = await manager.consolidate("t2")
        assert counts == {"extracted": 0, "merged": 0, "added": 0, "promoted": 0}
        assert len(provider.requests) == 1


class TestExtractionRobustness:
    async def test_garbage_extractor_output_is_tolerated(
        self, mem_store, embedder, emitter
    ):
        manager, _ = make_manager(
            mem_store, embedder, ["I could not find any facts, sorry!"],
            emitter=emitter,
        )
        seed_turn(mem_store, "session-1", "line")
        counts = await manager.consolidate("t1")
        assert counts["extracted"] == 0
        assert mem_store.fact_count() == 0

    async def test_fenced_json_is_parsed(self, mem_store, embedder):
        manager, _ = make_manager(
            mem_store, embedder,
            ['```json\n[{"text": "User prefers tabs", "type": "preference", '
             '"confidence": 0.9, "explicit": true, "quote": "always tabs"}]\n```'],
        )
        seed_turn(mem_store, "session-1", "always tabs")
        counts = await manager.consolidate("t1")
        assert counts["extracted"] == 1 and counts["promoted"] == 1

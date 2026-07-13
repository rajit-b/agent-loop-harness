"""Decay: idle facts archive at the floor; recall refreshes recency."""

from __future__ import annotations

from .conftest import make_manager


async def seed_active_fact(mem_store, embedder, text: str) -> int:
    [embedding] = await embedder.embed([text])
    fact_id = mem_store.insert_fact(
        text=text, type="preference", confidence=0.5, quote="",
        session_id="s0", turn_id="t0", embedding=embedding,
    )
    mem_store.set_status(fact_id, "active")
    return fact_id


class TestDecayArchiving:
    async def test_idle_fact_archives_below_floor(self, mem_store, embedder, now):
        # confidence 0.5, λ=0.01/day, floor 0.15 → archives after ~121 days
        manager, _ = make_manager(mem_store, embedder, [])
        fact_id = await seed_active_fact(mem_store, embedder, "User prefers tabs")
        now.advance_days(60)
        assert manager.decay_sweep() == []  # 0.5·e^-0.6 ≈ 0.27 — still alive
        now.advance_days(80)
        assert manager.decay_sweep() == [fact_id]  # 0.5·e^-1.4 ≈ 0.12 < 0.15
        assert mem_store.get_fact(fact_id).status == "archived"  # never deleted
        assert mem_store.fact_count() == 1

    async def test_archived_facts_are_excluded_from_recall(
        self, mem_store, embedder, now
    ):
        manager, _ = make_manager(mem_store, embedder, [])
        fact_id = await seed_active_fact(mem_store, embedder, "User prefers config tabs")
        now.advance_days(200)
        manager.decay_sweep()
        recalled = await manager.recall("config tabs preference?")
        assert recalled == []
        assert fact_id not in [r.fact.id for r in recalled]

    async def test_recall_refreshes_last_accessed_and_defers_decay(
        self, mem_store, embedder, now
    ):
        manager, _ = make_manager(mem_store, embedder, [])
        await seed_active_fact(mem_store, embedder, "User prefers config tabs")
        now.advance_days(100)
        recalled = await manager.recall("config tabs?")  # touch: resets the clock
        assert len(recalled) == 1
        now.advance_days(100)  # 200 idle days total, but only 100 since access
        assert manager.decay_sweep() == []  # survived thanks to the recall
        now.advance_days(50)
        assert len(manager.decay_sweep()) == 1  # eventually idles out

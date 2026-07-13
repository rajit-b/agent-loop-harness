"""MemoryManager: the loop's handle on all three tiers (§11).

- Working memory is the turn's message list — it lives in TurnContext,
  not here, and is never persisted beyond the trace.
- Short-term: session-scoped episodic log; when it exceeds
  memory.short_term.max_tokens the oldest half is folded into the rolling
  summary by a model call and the verbatim tail is kept.
- Long-term: consolidation extracts candidate facts from unprocessed
  episodic rows via a model call. Promotion: explicit directives
  immediately; otherwise extractor confidence ≥ threshold AND recurrence
  (≥ recurrence_min sessions, or > recurrence_min mentions overall).
  Dedup: cosine ≥ dedup_cosine against any existing fact merges instead
  of inserting. Decay: effective = confidence × exp(−λ·days idle);
  active facts under score_floor are ARCHIVED, never deleted. Recall
  refreshes last_accessed.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from agentloop.config.manifest import MemoryConfig
from agentloop.memory.store import FactRecord, MemoryStore
from agentloop.models.protocol import CompletionRequest, ModelProvider
from agentloop.types import (
    EmbeddingProvider,
    Message,
    NullEmitter,
    TraceEmitter,
)

_SECONDS_PER_DAY = 86_400.0
_DIRECTIVE = re.compile(
    r"\b(always|never|i prefer|we use|we always|we never|from now on|do not ever)\b",
    re.IGNORECASE,
)

SUMMARIZE_SYSTEM = (
    "You maintain a rolling summary of a conversation. Merge the previous "
    "summary with the new lines into one concise summary that preserves "
    "decisions, facts, and open questions. Reply with the summary only."
)

EXTRACT_SYSTEM = (
    "Extract durable facts about the user or project from this conversation "
    "segment: preferences, conventions, stable facts, entities. Respond with "
    "ONLY a JSON array; each item is {\"text\": str, \"type\": "
    "\"preference\"|\"fact\"|\"convention\"|\"entity\", \"confidence\": 0..1, "
    "\"explicit\": bool (true if the user stated it as a directive), "
    "\"quote\": the verbatim source sentence}. Reply [] if there are none."
)


@dataclass(frozen=True, slots=True)
class CandidateFact:
    text: str
    type: str
    confidence: float
    explicit: bool
    quote: str


@dataclass(frozen=True, slots=True)
class RecalledFact:
    fact: FactRecord
    similarity: float
    effective_score: float


def facts_block(recalled: Sequence[RecalledFact]) -> str:
    lines = ["## Known facts and preferences"]
    lines += [
        f"- [fact:{r.fact.id}] {r.fact.text} ({r.fact.type})" for r in recalled
    ]
    return "\n".join(lines)


class MemoryManager:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        provider: ModelProvider,
        *,
        config: MemoryConfig | None = None,
        session_id: str | None = None,
        recall_k: int = 5,
        emitter: TraceEmitter | None = None,
    ):
        self._store = store
        self._embedder = embedder
        self._provider = provider
        self._config = config or MemoryConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._recall_k = recall_k
        self._emitter = emitter or NullEmitter()

    # ------------------------------------------------------------------
    # short-term
    # ------------------------------------------------------------------

    def history_messages(self) -> list[Message]:
        """Rolling summary as one leading context message, then the
        verbatim unfolded tail (§11 injection point 2)."""
        messages: list[Message] = []
        summary = self._store.get_summary(self.session_id)
        if summary:
            messages.append(
                Message.user(f"[Summary of this session so far]\n{summary}")
            )
        for row in self._store.episodic_rows(self.session_id):
            if row.role == "assistant":
                messages.append(Message.assistant(row.content))
            else:
                messages.append(Message.user(row.content))
        return messages

    async def maybe_summarize(self) -> bool:
        rows = self._store.episodic_rows(self.session_id)
        total_tokens = sum(len(r.content) for r in rows) // 4
        if total_tokens <= self._config.short_term.max_tokens:
            return False
        oldest = rows[: max(1, len(rows) // 2)]  # oldest half folds
        lines = "\n".join(f"{r.role}: {r.content}" for r in oldest)
        previous = self._store.get_summary(self.session_id)
        result = await self._provider.complete(
            CompletionRequest(
                messages=(
                    Message.system(SUMMARIZE_SYSTEM),
                    Message.user(
                        f"Previous summary:\n{previous or '(none)'}\n\n"
                        f"New conversation lines:\n{lines}"
                    ),
                )
            )
        )
        self._store.set_summary(self.session_id, result.message.text.strip())
        self._store.fold_rows([r.id for r in oldest])
        self._emitter.emit(
            "memory.summarized",
            {"session_id": self.session_id, "folded_rows": len(oldest)},
        )
        return True

    # ------------------------------------------------------------------
    # long-term
    # ------------------------------------------------------------------

    async def recall(self, query: str, k: int | None = None) -> list[RecalledFact]:
        """Top-k active facts by similarity × decayed score; refreshes
        last_accessed on the returned facts."""
        limit = k or self._recall_k
        [query_vector] = await self._embedder.embed([query])
        candidates = self._store.knn_active_facts(query_vector, limit * 4)
        recalled = [
            RecalledFact(
                fact=fact,
                similarity=similarity,
                effective_score=self._effective_score(fact),
            )
            for fact, similarity in candidates
        ]
        recalled.sort(key=lambda r: r.similarity * r.effective_score, reverse=True)
        recalled = recalled[:limit]
        self._store.touch_facts([r.fact.id for r in recalled])
        if recalled:
            self._emitter.emit(
                "memory.recalled",
                {"count": len(recalled), "fact_ids": [r.fact.id for r in recalled]},
            )
        return recalled

    async def consolidate(self, turn_id: str = "") -> dict[str, int]:
        """Extract candidates from unprocessed episodic rows, then
        dedup-merge / insert / promote. Runs at on_turn_end (§11)."""
        cursor_key = f"consolidated_through:{self.session_id}"
        after_id = int(self._store.kv_get(cursor_key, "0"))
        rows = self._store.episodic_rows(
            self.session_id, include_folded=True, after_id=after_id
        )
        counts = {"extracted": 0, "merged": 0, "added": 0, "promoted": 0}
        if not rows:
            return counts

        candidates = await self._extract(rows)
        counts["extracted"] = len(candidates)
        long_term = self._config.long_term
        for candidate in candidates:
            [embedding] = await self._embedder.embed([candidate.text])
            nearest = self._store.nearest_fact(embedding)
            if nearest is not None and nearest[1] >= long_term.dedup_cosine:
                fact_id = nearest[0]
                self._store.merge_mention(
                    fact_id, self.session_id, candidate.confidence
                )
                counts["merged"] += 1
                self._emitter.emit("memory.fact_merged", {"fact_id": fact_id})
            else:
                fact_id = self._store.insert_fact(
                    text=candidate.text,
                    type=candidate.type,
                    confidence=candidate.confidence,
                    quote=candidate.quote,
                    session_id=self.session_id,
                    turn_id=turn_id,
                    embedding=embedding,
                )
                counts["added"] += 1
                self._emitter.emit("memory.fact_added", {"fact_id": fact_id})
            if self._should_promote(self._store.get_fact(fact_id), candidate):
                self._store.set_status(fact_id, "active")
                counts["promoted"] += 1
                self._emitter.emit(
                    "memory.fact_promoted",
                    {"fact_id": fact_id, "text": candidate.text},
                )
        self._store.kv_set(cursor_key, str(rows[-1].id))
        self.decay_sweep()
        return counts

    def decay_sweep(self) -> list[int]:
        """Archive active facts whose decayed score fell under the floor."""
        archived = []
        for fact in self._store.active_facts():
            if self._effective_score(fact) < self._config.long_term.score_floor:
                self._store.set_status(fact.id, "archived")
                archived.append(fact.id)
                self._emitter.emit("memory.fact_archived", {"fact_id": fact.id})
        return archived

    # ------------------------------------------------------------------

    async def on_turn_end(
        self, *, user_text: str, assistant_text: str, turn_id: str
    ) -> None:
        self._store.append_episodic(self.session_id, turn_id, "user", user_text)
        if assistant_text:
            self._store.append_episodic(
                self.session_id, turn_id, "assistant", assistant_text
            )
        await self.maybe_summarize()
        await self.consolidate(turn_id)

    # ------------------------------------------------------------------

    def _effective_score(self, fact: FactRecord) -> float:
        idle_days = max(0.0, (self._store._now() - fact.last_accessed)) / _SECONDS_PER_DAY  # noqa: SLF001
        return fact.confidence * math.exp(
            -self._config.long_term.decay_lambda * idle_days
        )

    def _should_promote(self, fact: FactRecord, candidate: CandidateFact) -> bool:
        if fact.status != "candidate":
            return False
        if candidate.explicit or _DIRECTIVE.search(candidate.quote or candidate.text):
            return True
        long_term = self._config.long_term
        if fact.confidence < long_term.promotion_confidence:
            return False
        return (
            fact.session_count >= long_term.recurrence_min
            or fact.support_count > long_term.recurrence_min
        )

    async def _extract(self, rows) -> list[CandidateFact]:
        transcript = "\n".join(f"{r.role}: {r.content}" for r in rows)
        result = await self._provider.complete(
            CompletionRequest(
                messages=(Message.system(EXTRACT_SYSTEM), Message.user(transcript))
            )
        )
        return self._parse_candidates(result.message.text)

    def _parse_candidates(self, text: str) -> list[CandidateFact]:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return []
        try:
            raw = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            self._emitter.emit("memory.extract_failed", {"snippet": text[:200]})
            return []
        candidates = []
        for item in raw:
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            candidates.append(
                CandidateFact(
                    text=str(item["text"]).strip(),
                    type=str(item.get("type", "fact")),
                    confidence=float(item.get("confidence", 0.5)),
                    explicit=bool(item.get("explicit", False)),
                    quote=str(item.get("quote", "")),
                )
            )
        return candidates

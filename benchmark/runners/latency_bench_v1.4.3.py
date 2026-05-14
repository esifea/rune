#!/usr/bin/env python3
"""Rune × envector-msa-1.4.3 latency benchmark.

Measures wall-clock latency of each rune pipeline phase broken down by
pipeline phase.  Runs standalone — no MCP server needed; adapters are imported
directly.

Differences from v1.2.2 benchmark:
  - eval_mode: mm32  (v1.2.2 was rmp)
  - index_type: ivf_vct  (v1.2.2 was flat)
  - insert_mode: single | batch
      single — one index.insert(data=[vec]) call per vector
      batch  — one index.insert(data=[v1,...,vN]) call per batch_size vectors

Scenarios (all target ivf_vct index, eval_mode=mm32)
----------
  capture:
    T1  Short English text  (~30 tokens)
    T2  Long English text   (~150 tokens)
    T3  Korean text
    T4  Duplicate input     (novelty near-duplicate path)
  recall:
    T5  Exact match query
    T6  Cross-language semantic query (Korean -> English)
    T7  topk scaling        (topk = 1, 3, 5, 10)
  vault_status:
    T8  Vault health check + diagnostics
  multi_capture:
    T13 2-phase batch embed+insert
    T14 5-phase batch embed+insert
  searchable:
    (3-phase decomposition of "capture → true SEARCHABLE (`done=true`)",
     aligned with the SDK 1.4.3 + cluster lifecycle (raw → merged → published):
       insert_rpc   — Index.insert(execute_until="segmentation",
                      await_completion=False, load=False) — split submission
                      plus auto-queued async_merge_by_request_ids, returns
                      as soon as data lands as raw shards.
       merge_wait   — Indexer.wait_for_index_operations_state(
                      request_ids, target_state=MERGED_SAVED=6): poll until
                      raw shards are merged.
       publish_wait — Index.load() (now safe — merged shards exist) +
                      Indexer.wait_for_index_operations_state(
                      request_ids, target_state=SEARCHABLE=7): poll until
                      done=true (true SEARCHABLE=7 in proto).
     `Index.load()` is also called once before measurement starts to ensure
     the index is loaded — calling it after insert while raw shards still
     exist triggers ForwardLoadRawShard which the cluster does not support.
     SDK's SEARCHABLE wait short-circuits as soon as the server reports `done=true`
     regardless of the reported state. Requires pyenvector 1.4.x.)
    T10 Short English
    T11 Long English
    T12 Korean

Modes
-----
  default (production index)
      Delegate to Rune-Vault for keys/credentials and run against the live
      runecontext index. No reset is performed.

  --direct-envector (benchmark index)
      Vault is still used for keys, score decryption, and metadata DEK.
      This flag will:
        * override the bundle's `index_name` with a dedicated bench index
          (default `runecontext_bench`) so the bench can never touch the
          live data
        * provision that bench index with the same params as production
          (IVF_VCT, nlist=256, default_nprobe=6, dim=1024, mm32, plain
          query encryption, no metadata encryption)
        * drop + recreate that bench index between scenarios so each
          scenario's latency numbers start on a clean status
        * use `auto_key_setup=False` to avoid the SDK's
          trying to unload `vault-key` while the live runecontext
          index still references it
        * reuse `vault-key` (same key the production index uses)

Usage
-----
  python benchmark/runners/latency_bench_v1.4.3.py --insert-mode single
  python benchmark/runners/latency_bench_v1.4.3.py --insert-mode batch
  python benchmark/runners/latency_bench_v1.4.3.py \\
      --insert-mode single --feature capture --runs 5

  # Bench-index mode with per-scenario reset:
  python benchmark/runners/latency_bench_v1.4.3.py \\
      --insert-mode single --direct-envector --feature searchable --runs 7

  python benchmark/runners/latency_bench_v1.4.3.py \\
      --insert-mode single --runs 10 --warmup 2 \\
      --report benchmark/reports/latency_results_v1.4.3_ivfvct_single_2026-05-11.md
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RUNE_DIR = BENCHMARK_DIR.parent
MCP_DIR = RUNE_DIR / "mcp"

for _p in (str(RUNE_DIR), str(MCP_DIR), str(BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runners.common import (  # noqa: E402
    REPORTS_DIR,
    LatencyBenchReport,
    LatencyScenarioResult,
    PhaseLatency,
)

# ── constants ─────────────────────────────────────────────────────────────────

EVAL_MODE = "mm32"
INDEX_TYPE = "ivf_vct"

# Direct-envector bench-index params used only when --direct-envector is set
BENCH_DIM = 1024
BENCH_INDEX_PARAMS = {"index_type": "IVF_VCT", "nlist": 256, "default_nprobe": 6}

# ── sample inputs ─────────────────────────────────────────────────────────────

_SHORT_TEXT = (
    "We decided to use PostgreSQL as our primary database. "
    "Team familiarity and mature ecosystem were the key reasons. "
    "Redis considered but rejected due to durability concerns."
)

_LONG_TEXT = (
    "Full ADR: Context — our monolith hit 10k RPS limits. "
    "Considered: (1) horizontal scale with read replicas, "
    "(2) CQRS split, (3) microservices decomposition. "
    "Trade-offs: read replicas cheapest but doesn't solve write bottleneck; "
    "CQRS complex but keeps single codebase; microservices highest ops overhead. "
    "Decision: CQRS with event sourcing on order-service first. "
    "Rationale: allows independent scaling of read path, event log gives audit "
    "trail for compliance (legal requirement from Q2 review). "
    "Rollback plan: feature flag, revert within 2 sprints if P99 > 500ms."
)

_KOREAN_TEXT = (
    "Redis를 캐시 레이어로 사용하기로 결정했습니다. "
    "Memcached도 검토했지만 데이터 구조 지원(Sorted Set, List)이 필요해서 Redis로 확정. "
    "TTL은 1시간으로 설정. 담당: 백엔드팀."
)

SCENARIOS_CAPTURE = [
    {
        "id": "T1_short_en",
        "text": _SHORT_TEXT,
        "title": "PostgreSQL chosen as primary DB",
        "domain": "architecture",
        "metadata": {"label": "short English (~30 tokens)", "tokens_approx": 35},
    },
    {
        "id": "T2_long_en",
        "text": _LONG_TEXT,
        "title": "CQRS event sourcing on order service",
        "domain": "architecture",
        "metadata": {"label": "long English (~150 tokens)", "tokens_approx": 155},
    },
    {
        "id": "T3_korean",
        "text": _KOREAN_TEXT,
        "title": "Redis 캐시 레이어 결정",
        "domain": "architecture",
        "metadata": {"label": "Korean text", "tokens_approx": 50},
    },
]

SCENARIOS_RECALL = [
    {
        "id": "T5_exact_match",
        "query": "Why did we choose PostgreSQL?",
        "topk": 5,
        "metadata": {"label": "exact match query"},
    },
    {
        "id": "T6_cross_lang",
        "query": "데이터베이스 선택 이유",
        "topk": 5,
        "metadata": {"label": "cross-language semantic (KO→EN)"},
    },
]

RECALL_TOPK_VARIANTS = [1, 3, 5, 10]

_MULTI_2_PHASE = [
    (
        "We chose PostgreSQL as the primary database. "
        "Team familiarity and mature ecosystem were decisive factors."
    ),
    (
        "Redis selected for session caching layer. "
        "TTL set to 30 minutes. Memcached rejected due to lack of data structure support."
    ),
]

_MULTI_5_PHASE = [
    (
        "Context: monolith hit 10k RPS ceiling. "
        "Decision: migrate to event-driven microservices architecture."
    ),
    (
        "Auth service extracted first. OAuth2 with JWT chosen. "
        "Session cookies rejected for statelessness requirement."
    ),
    (
        "Order service adopts CQRS. Write path via Kafka, "
        "read path via PostgreSQL read replicas."
    ),
    (
        "API gateway with per-tenant rate limiting (1000 req/s). "
        "Nginx selected over custom solution for operational maturity."
    ),
    (
        "Deployment on Kubernetes with Helm charts. "
        "Blue-green strategy for zero-downtime releases. Rollback within one sprint."
    ),
]

SCENARIOS_MULTI_CAPTURE = [
    {
        "id": "T13_multi_2phase",
        "texts": _MULTI_2_PHASE,
        "domain": "architecture",
        "metadata": {"label": "2-phase multi-capture", "phase_count": 2},
    },
    {
        "id": "T14_multi_5phase",
        "texts": _MULTI_5_PHASE,
        "domain": "architecture",
        "metadata": {"label": "5-phase multi-capture", "phase_count": 5},
    },
]


# ── timing helper ─────────────────────────────────────────────────────────────

class _Timer:
    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ── benchmark class ───────────────────────────────────────────────────────────

class LatencyBenchmark:
    """
    Latency benchmark for envector-msa-1.4.3 (eval_mode=mm32, index_type=ivf_vct).

    insert_mode controls how vectors are submitted during capture scenarios:
      "single" — index.insert(data=[vec]) called once per vector
      "batch"  — index.insert(data=[v1,...,vN]) called once per batch
    """

    def __init__(
        self,
        runs: int = 10,
        warmup: int = 2,
        insert_mode: str = "single",
        direct_envector: bool = False,
        bench_index_name: str = "runecontext_bench",
    ) -> None:
        self.runs = runs
        self.warmup = warmup
        self.insert_mode = insert_mode
        self.direct_envector = direct_envector
        self.bench_index_name = bench_index_name
        self._config: Any = None
        self._index_name: Optional[str] = None
        self._key_id: Optional[str] = None
        self._embedding: Any = None
        self._ev_client: Any = None
        self._vault: Any = None
        self._agent_dek: Optional[bytes] = None

    # ── setup ─────────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        from agents.common.config import load_config

        cfg = load_config()
        self._config = cfg

        if self.direct_envector:
            await self._setup_direct_envector()
        else:
            await self._setup_vault()

    async def _setup_vault(self) -> None:
        from agents.common.embedding_service import EmbeddingService
        from agents.common.envector_client import EnVectorClient
        from adapter.vault_client import VaultClient

        cfg = self._config

        print("  Connecting to Vault …", end=" ", flush=True)
        vault = VaultClient(
            vault_endpoint=cfg.vault.endpoint,
            vault_token=cfg.vault.token,
            ca_cert=cfg.vault.ca_cert or None,
            tls_disable=cfg.vault.tls_disable,
        )

        bundle = await vault.get_public_key()

        key_id = bundle.pop("key_id", None)
        index_name = bundle.pop("index_name", None)
        agent_id = bundle.pop("agent_id", None)
        agent_dek_b64 = bundle.pop("agent_dek", None)
        ev_endpoint = bundle.pop("envector_endpoint", None) or cfg.envector.endpoint
        ev_api_key = bundle.pop("envector_api_key", None) or cfg.envector.api_key
        ev_secure = bundle.pop("envector_secure", None)
        if ev_secure is None:
            ev_secure = cfg.envector.secure

        if not key_id:
            raise RuntimeError("Vault did not return key_id")
        if not index_name:
            raise RuntimeError("Vault did not return index_name")

        key_path = Path.home() / ".rune" / "keys"
        key_dir = key_path / key_id
        key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for filename, content in bundle.items():
            fp = key_dir / filename
            fd = os.open(str(fp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(content)

        agent_dek: Optional[bytes] = None
        if agent_dek_b64:
            agent_dek = base64.b64decode(agent_dek_b64)

        self._index_name = index_name
        self._key_id = key_id
        self._vault = vault
        self._agent_dek = agent_dek

        self._embedding = EmbeddingService(
            mode=cfg.embedding.mode,
            model=cfg.embedding.model,
        )

        self._ev_client = EnVectorClient(
            address=ev_endpoint,
            key_path=str(key_path),
            key_id=key_id,
            access_token=ev_api_key,
            secure=ev_secure,
            auto_key_setup=False,
            agent_id=agent_id,
            agent_dek=agent_dek,
            eval_mode=EVAL_MODE,
            index_type=INDEX_TYPE,
        )

        print("OK")
        print(f"    index      : {index_name}")
        print(f"    key_id     : {key_id}")
        print(f"    endpoint   : {ev_endpoint}")
        print(f"    eval_mode  : {EVAL_MODE}")
        print(f"    index_type : {INDEX_TYPE}")
        print(f"    insert_mode: {self.insert_mode}")

    async def _setup_direct_envector(self) -> None:
        """Benchmark index mode

        Connects to Vault the same way the production path does: to pick up
        `vault-key`, `agent_dek`, and the envector credentials
        But create dedicated benchmark-only index that can be dropped and recreated
        between scenarios without touching the live runecontext data.

        We keep using Vault for FHE score decryption (the SecKey lives on Vault.
        """
        from agents.common.embedding_service import EmbeddingService
        from agents.common.envector_client import EnVectorClient
        from adapter.vault_client import VaultClient

        cfg = self._config

        print("  Connecting to Vault (for benchmark index mode)...", end=" ", flush=True)
        vault = VaultClient(
            vault_endpoint=cfg.vault.endpoint,
            vault_token=cfg.vault.token,
            ca_cert=cfg.vault.ca_cert or None,
            tls_disable=cfg.vault.tls_disable,
        )
        bundle = await vault.get_public_key()

        key_id = bundle.pop("key_id", None)
        bundle.pop("index_name", None)
        agent_id = bundle.pop("agent_id", None)
        agent_dek_b64 = bundle.pop("agent_dek", None)
        ev_endpoint = bundle.pop("envector_endpoint", None) or cfg.envector.endpoint
        ev_api_key = bundle.pop("envector_api_key", None) or cfg.envector.api_key
        ev_secure = bundle.pop("envector_secure", None)
        if ev_secure is None:
            ev_secure = cfg.envector.secure

        if not key_id:
            raise RuntimeError("Vault did not return key_id")

        key_path = Path.home() / ".rune" / "keys"
        key_dir = key_path / key_id
        key_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for filename, content in bundle.items():
            fp = key_dir / filename
            fd = os.open(str(fp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(content)

        agent_dek: Optional[bytes] = None
        if agent_dek_b64:
            agent_dek = base64.b64decode(agent_dek_b64)

        self._index_name = self.bench_index_name
        self._key_id = key_id
        self._vault = vault
        self._agent_dek = agent_dek

        self._embedding = EmbeddingService(
            mode=cfg.embedding.mode,
            model=cfg.embedding.model,
        )

        # auto_key_setup=False so ev.init() doesn't trip the unload-others
        # We must not try to unload existing vault-key
        self._ev_client = EnVectorClient(
            address=ev_endpoint,
            key_path=str(key_path),
            key_id=key_id,
            access_token=ev_api_key,
            secure=ev_secure,
            auto_key_setup=False,
            agent_id=agent_id,
            agent_dek=agent_dek,
            eval_mode=EVAL_MODE,
            index_type=INDEX_TYPE,
        )

        last_err = None
        for _attempt in range(5):
            try:
                self._ev_client._ensure_initialized()
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2.0)
        if last_err is not None:
            raise last_err

        # Clean start
        self._reset_bench_index()

        print("OK")
        print(f"    index      : {self._index_name}  (bench-only, separate from runecontext)")
        print(f"    key_id     : {self._key_id}  (shared with production - read-only here)")
        print(f"    endpoint   : {ev_endpoint}")
        print(f"    eval_mode  : {EVAL_MODE}")
        print(f"    index_type : {INDEX_TYPE}")
        print(f"    insert_mode: {self.insert_mode}")
        print(f"    reset      : per-scenario drop+create")

    def _reset_bench_index(self) -> None:
        if not self.direct_envector:
            raise RuntimeError(
                "_reset_bench_index is bench-index mode only - refusing to "
                "drop the production index."
            )
        if self._index_name == "runecontext":
            raise RuntimeError(
                f"_reset_bench_index refusing to operate on production index "
                f"name 'runecontext' - set --bench-index to something else."
            )

        import pyenvector as ev

        adapter = self._ev_client._adapter

        def _do_reset():
            existing = ev.get_index_list()
            existing_names: list[str] = []
            if hasattr(existing, "indexes"):
                existing_names = [idx.index_name for idx in existing.indexes]
            elif isinstance(existing, (list, tuple)):
                existing_names = [str(idx) for idx in existing]

            if self._index_name in existing_names:
                ev.drop_index(self._index_name)
            ev.create_index(
                index_name=self._index_name,
                dim=BENCH_DIM,
                index_params=BENCH_INDEX_PARAMS,
                query_encryption="plain",
                metadata_encryption=False,
                metadata_key=b"",
            )

        adapter._with_reconnect(_do_reset)

    async def teardown(self) -> None:
        if self._vault is not None:
            await self._vault.close()

    def _warmup_label(self, run_i: int) -> str:
        return "warmup" if run_i < self.warmup else f"run {run_i - self.warmup + 1}"

    def _build_phase_list(
        self,
        phase_names: list[str],
        all_timings: list[dict[str, float]],
    ) -> list[PhaseLatency]:
        valid = all_timings[self.warmup:]
        phases = []
        for name in phase_names:
            samples = [t[name] for t in valid if name in t]
            phases.append(PhaseLatency(name=name, samples_ms=samples))
        return phases

    def _build_insert_metadata(self, text: str, title: str, domain: str) -> dict:
        return {
            "id": f"bench-{domain}-{int(time.time()*1000)}",
            "title": title,
            "domain": domain,
            "status": "accepted",
            "reusable_insight": text[:120],
            "payload": {"text": text},
            "why": {"certainty": "supported"},
        }

    # ── single capture phases ─────────────────────────────────────────────────

    async def _single_capture_phases(
        self, text: str, title: str, domain: str, batch_size: int = 1
    ) -> dict[str, float]:
        """
        Run one capture iteration and return per-phase latencies (ms).

        insert_mode=single: use_row_insert=True,  data=[vec]            — row insert API
        insert_mode=batch:  use_row_insert=False, data=[vec]*batch_size — batch insert API

        Phases: embed / score / vault_topk / insert / total
        """
        reusable_insight = text[:120]
        total_start = time.perf_counter()

        # [1] Embed
        with _Timer() as t_embed:
            vec = self._embedding.embed_single(reusable_insight)
        embed_ms = t_embed.elapsed_ms

        # [2] Novelty score (encrypted similarity search)
        with _Timer() as t_score:
            score_res = self._ev_client.score(self._index_name, vec)
        score_ms = t_score.elapsed_ms

        # [3] Vault TopK decrypt
        vault_ms = 0.0
        blobs = score_res.get("encrypted_blobs", []) if score_res.get("ok") else []
        if blobs:
            with _Timer() as t_vault:
                await self._vault.decrypt_search_results(blobs[0], top_k=3)
            vault_ms = t_vault.elapsed_ms

        # [4] Insert
        if self.insert_mode == "batch" and batch_size > 1:
            vectors = [vec] * batch_size
            metadata = [self._build_insert_metadata(text, title, domain) for _ in range(batch_size)]
        else:
            vectors = [vec]
            metadata = [self._build_insert_metadata(text, title, domain)]

        use_row = self.insert_mode == "single"
        with _Timer() as t_insert:
            self._ev_client.insert(
                index_name=self._index_name,
                vectors=vectors,
                metadata=metadata,
                use_row_insert=use_row,
            )
        insert_ms = t_insert.elapsed_ms

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "embed": embed_ms,
            "score": score_ms,
            "vault_topk": vault_ms,
            "insert": insert_ms,
            "total": total_ms,
        }

    # ── recall phases ─────────────────────────────────────────────────────────

    async def _single_recall_phases(
        self, query: str, topk: int
    ) -> dict[str, float]:
        """Phases: embed / score / vault_topk / remind / total"""
        total_start = time.perf_counter()

        with _Timer() as t_embed:
            vec = self._embedding.embed_single(query)
        embed_ms = t_embed.elapsed_ms

        with _Timer() as t_score:
            score_res = self._ev_client.score(self._index_name, vec)
        score_ms = t_score.elapsed_ms

        vault_ms = 0.0
        remind_ms = 0.0
        blobs = score_res.get("encrypted_blobs", []) if score_res.get("ok") else []
        if blobs:
            with _Timer() as t_vault:
                vault_res = await self._vault.decrypt_search_results(blobs[0], top_k=topk)
            vault_ms = t_vault.elapsed_ms

            if vault_res.ok and vault_res.results:
                with _Timer() as t_remind:
                    self._ev_client.remind(
                        self._index_name,
                        vault_res.results,
                        output_fields=["metadata"],
                    )
                remind_ms = t_remind.elapsed_ms

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "embed": embed_ms,
            "score": score_ms,
            "vault_topk": vault_ms,
            "remind": remind_ms,
            "total": total_ms,
        }

    # ── scenario runners ───────────────────────────────────────────────────────

    async def run_capture_scenario(self, scenario: dict) -> LatencyScenarioResult:
        sid = scenario["id"]
        text = scenario["text"]
        title = scenario["title"]
        domain = scenario["domain"]
        meta = scenario.get("metadata", {})

        print(f"  [{sid}] ", end="", flush=True)
        all_timings: list[dict[str, float]] = []

        for i in range(self.runs):
            label = self._warmup_label(i)
            print(f"{label} ", end="", flush=True)
            try:
                t = await self._single_capture_phases(text, title, domain)
                all_timings.append(t)
            except Exception as e:
                print(f"\n    ERROR on {label}: {e}")
                return LatencyScenarioResult(
                    scenario_id=sid,
                    feature="capture",
                    metadata=meta,
                    error=str(e),
                )

        print("done")
        phases = self._build_phase_list(
            ["embed", "score", "vault_topk", "insert", "total"],
            all_timings,
        )
        return LatencyScenarioResult(
            scenario_id=sid,
            feature="capture",
            phases=phases,
            metadata={**meta, "insert_mode": self.insert_mode, "runs": self.runs - self.warmup},
        )

    async def run_capture_duplicate(self) -> LatencyScenarioResult:
        """T4: capture same text twice — second hits near-duplicate path."""
        sid = "T4_duplicate"
        text = _SHORT_TEXT
        title = "PostgreSQL chosen as primary DB (duplicate)"
        meta = {"label": "duplicate input — tests novelty near-duplicate path"}

        print(f"  [{sid}] ", end="", flush=True)
        all_timings: list[dict[str, float]] = []

        try:
            await self._single_capture_phases(text, title, "architecture")
        except Exception:
            pass

        for i in range(self.runs):
            label = self._warmup_label(i)
            print(f"{label} ", end="", flush=True)
            try:
                t = await self._single_capture_phases(text, title, "architecture")
                all_timings.append(t)
            except Exception as e:
                print(f"\n    ERROR: {e}")
                return LatencyScenarioResult(
                    scenario_id=sid, feature="capture", metadata=meta, error=str(e)
                )

        print("done")
        phases = self._build_phase_list(
            ["embed", "score", "vault_topk", "insert", "total"],
            all_timings,
        )
        return LatencyScenarioResult(
            scenario_id=sid,
            feature="capture",
            phases=phases,
            metadata={**meta, "insert_mode": self.insert_mode, "runs": self.runs - self.warmup},
        )

    async def run_recall_scenario(self, scenario: dict) -> LatencyScenarioResult:
        sid = scenario["id"]
        query = scenario["query"]
        topk = scenario.get("topk", 5)
        meta = scenario.get("metadata", {})

        print(f"  [{sid}] ", end="", flush=True)
        all_timings: list[dict[str, float]] = []

        for i in range(self.runs):
            label = self._warmup_label(i)
            print(f"{label} ", end="", flush=True)
            try:
                t = await self._single_recall_phases(query, topk)
                all_timings.append(t)
            except Exception as e:
                print(f"\n    ERROR: {e}")
                return LatencyScenarioResult(
                    scenario_id=sid, feature="recall", metadata=meta, error=str(e)
                )

        print("done")
        phases = self._build_phase_list(
            ["embed", "score", "vault_topk", "remind", "total"],
            all_timings,
        )
        return LatencyScenarioResult(
            scenario_id=sid,
            feature="recall",
            phases=phases,
            metadata={**meta, "topk": topk, "runs": self.runs - self.warmup},
        )

    async def run_recall_topk_scaling(self) -> list[LatencyScenarioResult]:
        """T7: measure recall latency at varying topk values."""
        results = []
        query = "architecture decisions"

        for topk in RECALL_TOPK_VARIANTS:
            sid = f"T7_topk_{topk}"
            print(f"  [{sid}] ", end="", flush=True)
            all_timings: list[dict[str, float]] = []
            runs = max(self.warmup + 3, min(self.runs, self.warmup + 5))

            for i in range(runs):
                label = self._warmup_label(i)
                print(f"{label} ", end="", flush=True)
                try:
                    t = await self._single_recall_phases(query, topk)
                    all_timings.append(t)
                except Exception as e:
                    print(f"\n    ERROR: {e}")
                    results.append(LatencyScenarioResult(
                        scenario_id=sid, feature="recall",
                        metadata={"topk": topk, "label": f"topk scaling topk={topk}"},
                        error=str(e),
                    ))
                    break
            else:
                print("done")
                phases = self._build_phase_list(
                    ["embed", "score", "vault_topk", "remind", "total"],
                    all_timings,
                )
                results.append(LatencyScenarioResult(
                    scenario_id=sid,
                    feature="recall",
                    phases=phases,
                    metadata={"topk": topk, "label": f"topk scaling topk={topk}",
                              "runs": runs - self.warmup},
                ))
        return results

    async def _searchable_capture_phases(
        self, text: str, title: str, domain: str
    ) -> dict[str, float]:
        """
        Measure time from capture start until data is truly searchable
        (`done=true`) — 3-phase decomposition per SDK 1.4.3 lifecycle.

        Why we bypass the rune wrapper here:
          The rune wrapper's `await_searchable=True` maps to SDK
          `await_completion=True` + `execute_until="segmentation"`, which only
          reaches `MERGED_SAVED` (all items in non-raw shards but not yet
          published). To measure the true `SEARCHABLE` (`done=true`) latency,
          we drive the SDK `Index` directly through three explicit phases.

        Phases:
          embed              — embed locally
          score              — FHE novelty check
          vault_topk         — Vault decrypt
          insert_rpc         — Index.insert(execute_until="segmentation",
                               await_completion=False, load=False,
                               use_row_insert=<mode>, request_ids=[]):
                               split submission only. Returns as soon as data
                               lands as raw shards on the server. The
                               `request_ids` out-list is filled with the
                               server-generated request IDs that Phases B and C
                               poll against.
          merge_wait         — Indexer.wait_for_index_operations_state(
                               request_ids, target_state=MERGED_SAVED): polls
                               get_index_operation_status until each
                               request_id reaches MERGED_SAVED (=6 in proto).
                               This is the boundary where raw shards have
                               been merged into IVF clusters.
          publish_wait       — Index.load() + Indexer.wait_for_index_operations_state(
                               request_ids, target_state=SEARCHABLE): explicit
                               publication of the now-merged shards, then poll
                               until done=true (SEARCHABLE=7). The poll's
                               stop condition is `last.done == True` when
                               target_state == SEARCHABLE.
                               `load()` MUST happen after merge — calling it
                               earlier (while raw shards still exist) triggers
                               ForwardLoadRawShard, which the cluster does not
                               support. The index is also pre-loaded once
                               before measurement starts (setup) for the same
                               reason.
          total              — wall clock including all phases

        After `publish_wait`, runs a recall verification step OUTSIDE the
        measured latency window: re-runs the recall pipeline with the same
        vector and checks that the just-inserted record's unique id appears
        in top-10 results. If verification fails (e.g. publication did not
        actually expose the record), raises RuntimeError so the caller marks
        the scenario as FAIL — we want the "true SEARCHABLE" claim to be
        validated, not just trusted from `done=true`.

        Requires pyenvector 1.4.x in the venv. Will TypeError on 1.2.2
        (older `Index.insert` signature has only `data`/`metadata`).
        """
        import pyenvector as ev
        from pyenvector.proto_gen.v2.common.index_operation_message_pb2 import (
            IndexOperationState,
        )

        merged_state = IndexOperationState.Value("MERGED_SAVED")    # = 6
        searchable_state = IndexOperationState.Value("SEARCHABLE")  # = 7

        # Setup (outside measurement): ensure the index is loaded BEFORE we
        # insert. The cluster does not support ForwardLoadRawShard, so calling
        # Index.load() while raw shards are still pending (which is exactly
        # the state Phase A leaves the server in) crashes. Pre-loading here
        # is idempotent — the SDK treats "already loaded with no pending
        # shards" as a no-op.
        #
        # `_ensure_initialized()` itself has no retry; the first connect after
        # process start can flake on this cluster. Retry up to 5x so a cold
        # start doesn't fail the entire scenario before warmup runs.
        last_err = None
        for _attempt in range(5):
            try:
                self._ev_client._ensure_initialized()
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2.0)
        if last_err is not None:
            raise last_err
        adapter = self._ev_client._adapter
        def _ensure_index_loaded():
            idx = ev.Index(self._index_name)
            idx.load()
        adapter._with_reconnect(_ensure_index_loaded)

        reusable_insight = text[:120]
        total_start = time.perf_counter()

        with _Timer() as t_embed:
            vec = self._embedding.embed_single(reusable_insight)
        embed_ms = t_embed.elapsed_ms

        with _Timer() as t_score:
            score_res = self._ev_client.score(self._index_name, vec)
        score_ms = t_score.elapsed_ms

        vault_ms = 0.0
        blobs = score_res.get("encrypted_blobs", []) if score_res.get("ok") else []
        if blobs:
            with _Timer() as t_vault:
                await self._vault.decrypt_search_results(blobs[0], top_k=3)
            vault_ms = t_vault.elapsed_ms

        insert_metadata = self._build_insert_metadata(text, title, domain)
        expected_id = insert_metadata["id"]
        use_row = self.insert_mode == "single"
        request_ids: list[str] = []

        # The row-insert API path validates metadata as strings (server rejects
        # raw dicts with `async split data failed`). Mirror what
        # `invoke_insert` does: JSON-stringify, then app-encrypt with the
        # per-agent DEK if available. The batch path also accepts this format.
        metadata_str = [json.dumps(insert_metadata)]
        if adapter._agent_dek and adapter._agent_id:
            metadata_wire = [adapter._app_encrypt_metadata(m) for m in metadata_str]
        else:
            metadata_wire = metadata_str

        # [4] Phase A — insert RPC only. Data lands as raw shards on the server
        # and the call returns; merge/publication happen later in Phases B/C.
        with _Timer() as t_insert_rpc:
            def _do_insert():
                idx = ev.Index(self._index_name)
                idx.insert(
                    data=[vec],
                    metadata=metadata_wire,
                    await_completion=False,
                    load=False,
                    use_row_insert=use_row,
                    execute_until="segmentation",
                    request_ids=request_ids,
                )
            adapter._with_reconnect(_do_insert)
        insert_rpc_ms = t_insert_rpc.elapsed_ms

        # [5] Phase B — raw → merged: poll get_index_operation_status until
        # each request_id reaches MERGED_SAVED (=6). Since target is not
        # SEARCHABLE, the rank-based stop condition applies.
        # Phase A already submitted async_merge_by_request_ids via
        # `execute_until="segmentation"`, so the merge work is already in
        # flight by the time we start polling.
        with _Timer() as t_merge:
            def _do_merge_wait():
                idx = ev.Index(self._index_name)
                idx.indexer.wait_for_index_operations_state(
                    self._index_name,
                    request_ids,
                    target_state=merged_state,
                    timeout_s=120.0,
                    poll_interval_s=0.5,
                )
            adapter._with_reconnect(_do_merge_wait)
        merge_wait_ms = t_merge.elapsed_ms

        # [6] Phase C — merged → SEARCHABLE: publish then wait for done=true.
        # Poll get_index_operation_status until each request_id reports `done=true`.
        with _Timer() as t_publish:
            def _do_publish():
                idx = ev.Index(self._index_name)
                idx.load()
                idx.indexer.wait_for_index_operations_state(
                    self._index_name,
                    request_ids,
                    target_state=searchable_state,
                    timeout_s=120.0,
                    poll_interval_s=0.5,
                )
            adapter._with_reconnect(_do_publish)
        publish_wait_ms = t_publish.elapsed_ms

        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Recall verification — outside the measured latency window.
        # Re-runs the recall pipeline with the same vector and matches the
        # captured unique id against the decrypted metadata `id` field. If the
        # id is not found in top-10, raise — caller turns this into a scenario
        # FAIL via the existing try/except in run_searchable_scenario.
        await self._verify_searchable_recall(vec, expected_id)

        return {
            "embed": embed_ms,
            "score": score_ms,
            "vault_topk": vault_ms,
            "insert_rpc": insert_rpc_ms,
            "merge_wait": merge_wait_ms,
            "publish_wait": publish_wait_ms,
            "total": total_ms,
        }

    async def _verify_searchable_recall(
        self,
        vec: list,
        expected_id: str,
        top_k: int = 10,
    ) -> None:
        """Verify that the record we just inserted is actually recallable.

        Re-runs score → vault decrypt → remind with the same vector, then
        decrypts each result's app-layer-encrypted metadata locally with the
        per-agent DEK and checks whether `expected_id` appears among the
        top-`top_k` ids. Raises RuntimeError on any failure (missing DEK,
        empty results, decrypt failure, or id not found) so the caller treats
        the scenario as FAIL.
        """
        from pyenvector.utils.aes import decrypt_metadata as aes_decrypt

        if self._agent_dek is None:
            raise RuntimeError(
                "recall verification: agent_dek missing — cannot decrypt metadata"
            )

        score_res = self._ev_client.score(self._index_name, vec)
        if not score_res.get("ok"):
            raise RuntimeError(
                f"recall verification: score failed: {score_res.get('error')}"
            )
        blobs = score_res.get("encrypted_blobs", [])
        if not blobs:
            raise RuntimeError("recall verification: score returned no blobs")

        vault_res = await self._vault.decrypt_search_results(blobs[0], top_k=top_k)
        if not vault_res.ok or not vault_res.results:
            raise RuntimeError(
                f"recall verification: vault decrypt returned no results (ok={vault_res.ok})"
            )

        remind_res = self._ev_client.remind(
            self._index_name,
            vault_res.results,
            output_fields=["metadata"],
        )
        if not remind_res.get("ok"):
            raise RuntimeError(
                f"recall verification: remind failed: {remind_res.get('error')}"
            )

        found_ids: list[str] = []
        for entry in remind_res.get("results", []):
            meta_field = entry.get("metadata") or entry.get("data")
            if not isinstance(meta_field, str):
                continue
            try:
                wrapper = json.loads(meta_field)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(wrapper, dict):
                continue
            ct = wrapper.get("c")
            if not isinstance(ct, str):
                continue
            try:
                decrypted = aes_decrypt(ct, self._agent_dek)
            except Exception:
                continue
            if isinstance(decrypted, (bytes, bytearray)):
                try:
                    decrypted = json.loads(bytes(decrypted).decode())
                except Exception:
                    continue
            elif isinstance(decrypted, str):
                try:
                    decrypted = json.loads(decrypted)
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(decrypted, dict):
                mid = decrypted.get("id")
                if isinstance(mid, str):
                    found_ids.append(mid)

        if expected_id not in found_ids:
            raise RuntimeError(
                f"recall verification: expected id '{expected_id}' not found in "
                f"top-{top_k} results (got {len(found_ids)} decryptable ids)"
            )

    async def run_searchable_scenario(self, scenario: dict) -> LatencyScenarioResult:
        """T10-T12: measure capture → searchable latency.

        Total wall-clock = embed + score + vault_topk + insert_rpc + merge_wait
        + publish_wait. The last three phases drive the SDK 1.4.3 lifecycle
        directly to reach the true `SEARCHABLE` (`done=true`) state; see
        `_searchable_capture_phases` docstring for the rationale (bypassing the
        rune wrapper's `await_searchable` alias which only reaches `MERGED_SAVED`).
        """
        _parts = scenario["id"].split("_", 1)
        sid = f"T{int(_parts[0][1:]) + 9}_{_parts[1]}_searchable"
        text = scenario["text"]
        title = scenario["title"]
        domain = scenario["domain"]
        meta = scenario.get("metadata", {})

        print(f"  [{sid}] ", end="", flush=True)
        all_timings: list[dict[str, float]] = []

        for i in range(self.runs):
            label = self._warmup_label(i)
            print(f"{label} ", end="", flush=True)
            try:
                t = await self._searchable_capture_phases(text, title, domain)
                all_timings.append(t)
            except Exception as e:
                print(f"\n    ERROR on {label}: {e}")
                return LatencyScenarioResult(
                    scenario_id=sid,
                    feature="searchable",
                    metadata=meta,
                    error=str(e),
                )

        print("done")
        phases = self._build_phase_list(
            [
                "embed",
                "score",
                "vault_topk",
                "insert_rpc",
                "merge_wait",
                "publish_wait",
                "total",
            ],
            all_timings,
        )
        return LatencyScenarioResult(
            scenario_id=sid,
            feature="searchable",
            phases=phases,
            metadata={**meta, "runs": self.runs - self.warmup},
        )

    async def run_vault_status(self) -> LatencyScenarioResult:
        """T9: health check latency."""
        sid = "T9_vault_status"
        print(f"  [{sid}] ", end="", flush=True)
        samples: list[float] = []

        for i in range(self.runs):
            label = self._warmup_label(i)
            print(f"{label} ", end="", flush=True)
            with _Timer() as t:
                await self._vault.health_check()
            samples.append(t.elapsed_ms)

        print("done")
        valid = samples[self.warmup:]
        phases = [PhaseLatency(name="vault_health_check", samples_ms=valid)]
        return LatencyScenarioResult(
            scenario_id=sid,
            feature="vault_status",
            phases=phases,
            metadata={"label": "Vault gRPC health check", "runs": len(valid)},
        )

    # ── multi-phase capture ────────────────────────────────────────────────────

    async def _multi_capture_phases(
        self, texts: list[str], domain: str
    ) -> dict[str, float]:
        """
        Multi-phase capture: N records embedded and inserted as a batch.

        Mirrors the real capture path when a decision has multiple phases
        (server.py: record_builder.build_phases → insert_with_text(texts)).

        Phases:
          embed_batch  — embed(texts): single gRPC call, N vectors at once
          score        — novelty check on primary record (texts[0])
          vault_topk   — Vault decrypt on primary record's score
          insert_batch — insert all N vectors in one batch API call
          total        — wall clock including all phases
        """
        total_start = time.perf_counter()
        insights = [t[:120] for t in texts]

        # [1] Batch embed — uses embed(texts), not embed_single
        with _Timer() as t_embed:
            vecs = self._embedding.embed(insights)
        embed_ms = t_embed.elapsed_ms

        # [2] Novelty score on primary record (first phase)
        with _Timer() as t_score:
            score_res = self._ev_client.score(self._index_name, vecs[0])
        score_ms = t_score.elapsed_ms

        # [3] Vault TopK decrypt
        vault_ms = 0.0
        blobs = score_res.get("encrypted_blobs", []) if score_res.get("ok") else []
        if blobs:
            with _Timer() as t_vault:
                await self._vault.decrypt_search_results(blobs[0], top_k=3)
            vault_ms = t_vault.elapsed_ms

        # [4] Insert all N vectors as batch (use_row_insert=False)
        metadata = [
            self._build_insert_metadata(t, f"phase-{i + 1}", domain)
            for i, t in enumerate(texts)
        ]
        with _Timer() as t_insert:
            self._ev_client.insert(
                index_name=self._index_name,
                vectors=vecs,
                metadata=metadata,
                use_row_insert=False,
            )
        insert_ms = t_insert.elapsed_ms

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "embed_batch": embed_ms,
            "score": score_ms,
            "vault_topk": vault_ms,
            "insert_batch": insert_ms,
            "total": total_ms,
        }

    async def run_multi_capture_scenario(self, scenario: dict) -> LatencyScenarioResult:
        """T13-T14: multi-phase capture latency (batch embed + batch insert)."""
        sid = scenario["id"]
        texts = scenario["texts"]
        domain = scenario["domain"]
        meta = scenario.get("metadata", {})

        print(f"  [{sid}] ", end="", flush=True)
        all_timings: list[dict[str, float]] = []

        for i in range(self.runs):
            label = self._warmup_label(i)
            print(f"{label} ", end="", flush=True)
            try:
                t = await self._multi_capture_phases(texts, domain)
                all_timings.append(t)
            except Exception as e:
                print(f"\n    ERROR on {label}: {e}")
                return LatencyScenarioResult(
                    scenario_id=sid,
                    feature="multi_capture",
                    metadata=meta,
                    error=str(e),
                )

        print("done")
        phases = self._build_phase_list(
            ["embed_batch", "score", "vault_topk", "insert_batch", "total"],
            all_timings,
        )
        return LatencyScenarioResult(
            scenario_id=sid,
            feature="multi_capture",
            phases=phases,
            metadata={**meta, "runs": self.runs - self.warmup},
        )

    # ── network baseline ───────────────────────────────────────────────────────

    def _measure_network_rtt(self) -> str:
        host = (self._config.envector.endpoint or "").split(":")[0]
        if not host:
            return "unknown"
        try:
            out = subprocess.check_output(
                ["ping", "-c", "5", host],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode()
            for line in out.splitlines():
                if "avg" in line or "rtt" in line:
                    parts = line.split("=")[-1].strip().split("/")
                    if len(parts) >= 2:
                        return f"{parts[1]} ms (avg RTT)"
        except Exception:
            pass
        return "unknown"

    # ── orchestration ──────────────────────────────────────────────────────────

    async def run(
        self,
        feature_filter: Optional[str] = None,
    ) -> LatencyBenchReport:
        report = LatencyBenchReport()

        rtt = self._measure_network_rtt()
        cfg = self._config
        report.env = {
            "bench_version": "1.4.3",
            "date": __import__("datetime").date.today().isoformat(),
            "envector_endpoint": cfg.envector.endpoint,
            "vault_endpoint": cfg.vault.endpoint,
            "embedding_model": cfg.embedding.model,
            "embedding_mode": cfg.embedding.mode,
            "index_name": self._index_name,
            "key_id": self._key_id,
            "eval_mode": EVAL_MODE,
            "index_type": INDEX_TYPE,
            "insert_mode": self.insert_mode,
            "network_rtt": rtt,
            "runs_per_scenario": self.runs - self.warmup,
            "warmup_runs": self.warmup,
            "direct_envector": self.direct_envector,
            "reset_policy": (
                "per-scenario drop+create" if self.direct_envector
                else "no reset (shared production index)"
            ),
        }

        run_all = feature_filter is None
        run_capture = run_all or feature_filter == "capture"
        run_recall = run_all or feature_filter == "recall"
        run_vault = run_all or feature_filter == "vault_status"
        run_searchable = run_all or feature_filter == "searchable"
        run_multi = run_all or feature_filter == "multi_capture"

        # Only used with '--direct-envector' flag
        def _reset_for(scenario_label: str) -> None:
            if not self.direct_envector:
                return
            print(f"  reset[{scenario_label}]...", end=" ", flush=True)
            self._reset_bench_index()
            print("done")

        if run_capture:
            print("\n[capture]")
            for sc in SCENARIOS_CAPTURE:
                _reset_for(sc["id"])
                r = await self.run_capture_scenario(sc)
                report.add(r)
            _reset_for("T4_duplicate")
            r = await self.run_capture_duplicate()
            report.add(r)

        if run_recall:
            print("\n[recall]")
            for sc in SCENARIOS_RECALL:
                _reset_for(sc["id"])
                r = await self.run_recall_scenario(sc)
                report.add(r)
            _reset_for("T7_topk_scaling")
            for r in await self.run_recall_topk_scaling():
                report.add(r)

        if run_vault:
            print("\n[vault_status]")
            r = await self.run_vault_status()
            report.add(r)

        if run_searchable:
            print("\n[searchable]")
            for sc in SCENARIOS_CAPTURE[:3]:  # T1, T2, T3 — short/long/Korean
                _reset_for(sc["id"] + "_searchable")
                r = await self.run_searchable_scenario(sc)
                report.add(r)

        if run_multi:
            print("\n[multi_capture]")
            for sc in SCENARIOS_MULTI_CAPTURE:
                _reset_for(sc["id"])
                r = await self.run_multi_capture_scenario(sc)
                report.add(r)

        return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def _print_summary(report: LatencyBenchReport) -> None:
    print("\n" + "=" * 64)
    print(f"  rune latency benchmark — envector-msa-1.4.3 ({EVAL_MODE}/{INDEX_TYPE})")
    print("=" * 64)

    for s in report.scenarios:
        if s.error:
            print(f"  [FAIL] {s.scenario_id}: {s.error}")
            continue
        total_phase = next((p for p in s.phases if p.name == "total"), None)
        if total_phase and total_phase.samples_ms:
            print(
                f"  {s.scenario_id:<30} "
                f"p50={total_phase.p50:7.1f}ms  "
                f"p95={total_phase.p95:7.1f}ms  "
                f"n={total_phase.n}"
            )
        else:
            for p in s.phases:
                print(
                    f"  {s.scenario_id}/{p.name:<26} "
                    f"p50={p.p50:7.1f}ms  "
                    f"p95={p.p95:7.1f}ms  "
                    f"n={p.n}"
                )
    print("=" * 64 + "\n")


async def _main(args: argparse.Namespace) -> None:
    bench = LatencyBenchmark(
        runs=args.runs,
        warmup=args.warmup,
        insert_mode=args.insert_mode,
        direct_envector=args.direct_envector,
        bench_index_name=args.bench_index,
    )

    mode_label = "bench-index" if args.direct_envector else "vault-mediated"
    print(
        f"\nSetting up … (mode={mode_label}, eval_mode={EVAL_MODE}, "
        f"index_type={INDEX_TYPE}, insert_mode={args.insert_mode})"
    )
    await bench.setup()

    print(f"\nRunning benchmark (runs={args.runs - args.warmup} effective, warmup={args.warmup}) …")
    report = await bench.run(feature_filter=args.feature)

    await bench.teardown()

    _print_summary(report)

    if args.report:
        report_path = Path(args.report)
        if args.format == "json":
            saved = report.save_json(report_path)
        else:
            saved = report.save_markdown(report_path)
        print(f"Report saved → {saved}")
    else:
        print(report.to_markdown())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Rune × envector-msa-1.4.3 latency benchmark ({EVAL_MODE}/{INDEX_TYPE})"
    )
    parser.add_argument(
        "--insert-mode",
        choices=["single", "batch"],
        required=True,
        help="Insert mode: single (one vector per call) or batch (N vectors per call)",
    )
    parser.add_argument(
        "--feature",
        choices=["capture", "recall", "vault_status", "searchable", "multi_capture"],
        default=None,
        help="Run only this feature (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Total runs per scenario including warmup (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs to discard (default: 2)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to save the report (default: print to stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Report format (default: md)",
    )
    parser.add_argument(
        "--direct-envector",
        action="store_true",
        help=(
            "Benchmark index mode: use a dedicated bench index instead of the "
            "live index, drop+recreate it between scenarios for clean "
            "latency numbers. Vault is still used for keys and FHE score "
            "decryption (the SecKey only lives on Vault). "
            "Does NOT touch the live data."
        ),
    )
    parser.add_argument(
        "--bench-index",
        default="runecontext_bench",
        help="Bench index name (--direct-envector only, default: runecontext_bench)",
    )
    args = parser.parse_args()

    if args.warmup >= args.runs:
        parser.error(f"--warmup ({args.warmup}) must be < --runs ({args.runs})")

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()

"""End-to-end episode orchestration (Phases 2-6).

run_episode() wires: GraphRAG retrieval -> heuristic scoring -> LLM planning ->
deterministic validation (safety guard) -> MuJoCo execution (only if validation
passes) -> world-state update -> episode logging.

This is the single entry point the evaluation harness and demos call. It is the
concrete realization of the project's safety stance: nothing executes unless the
deterministic validator says ok.
"""

from __future__ import annotations

import uuid
from typing import Optional, Tuple

from r4r.embeddings.base import Embedder
from r4r.logging_.episode import EpisodeLog
from r4r.logging_.logger import EpisodeLogger
from r4r.planner.backends import LLMBackend
from r4r.planner.context import build_context
from r4r.planner.planner import Planner
from r4r.retrieval.retriever import GraphRAGRetriever
from r4r.scene_graph.graph import SceneGraph
from r4r.scoring.heuristics import score_scene
from r4r.validation.validator import PlanValidator


def _failure_category(plan_valid, plan_fallback, val_ok, executed, exec_ok) -> Optional[str]:
    if not plan_valid or plan_fallback:
        return "plan_refused"
    if not val_ok:
        return "validation_blocked"
    if not executed:
        return "not_executed"
    if not exec_ok:
        return "execution_failed"
    return None


def run_episode(
    graph: SceneGraph,
    instruction: str,
    embedder: Embedder,
    backend: LLMBackend,
    logger: Optional[EpisodeLogger] = None,
    execute: bool = True,
    k_entities: int = 4,
    depth: int = 1,
    execution_backend: str = "kinematic",
    on_exec_step=None,
) -> Tuple[EpisodeLog, SceneGraph]:
    """execution_backend: 'kinematic' (MJCF built from the graph, attach-and-carry)
    or 'piper' (real AgileX Piper arm + IK; the graph must mirror the Piper
    MuJoCo scene — see build_piper_scene_graph). on_exec_step(executor) is an
    optional per-physics-step hook for viewers/recorders."""
    # 1. retrieve
    retriever = GraphRAGRetriever(graph, embedder).build()
    res = retriever.retrieve(instruction, k_entities=k_entities, depth=depth)

    # 2. score
    report = score_scene(graph)

    # 3. plan
    ctx = build_context(instruction, res.serialized_subgraph, res.knowledge_chunks, report, graph)
    plan_result = Planner(backend).plan(ctx)

    # 4. validate (safety guard)
    validation = PlanValidator().validate(plan_result.plan, graph, ctx.constraints, report)

    log = EpisodeLog(
        episode_id=uuid.uuid4().hex[:12],
        instruction=instruction,
        subgraph_hash=res.serialized_subgraph.get("graph_hash", ""),
        subgraph_summary={"nodes": len(res.subgraph), "edges": len(res.subgraph.edges())},
        retrieval_seeds=res.seed_ids,
        knowledge_ids=[c["id"] for c in res.knowledge_chunks],
        token_estimate=res.token_estimate,
        full_graph_token_estimate=res.full_graph_token_estimate,
        scores=report.compact(),
        plan=plan_result.plan.model_dump(),
        plan_valid=plan_result.valid,
        plan_used_fallback=plan_result.used_fallback,
        plan_attempts=plan_result.attempts,
        planner_backend=plan_result.backend,
        validation_ok=validation.ok,
        validation_violations=[v.model_dump() for v in validation.violations],
    )

    # 5. execute ONLY if the safety guard passed
    executed = False
    exec_ok = False
    if execute and validation.ok:
        from r4r.sim.executor import update_graph_from_execution
        if execution_backend == "piper":
            from r4r.sim.piper import PiperArmExecutor
            executor = PiperArmExecutor()
        else:
            from r4r.sim.executor import MujocoExecutor
            executor = MujocoExecutor(graph)
        if on_exec_step is not None:
            executor.on_step = on_exec_step
        exec_result = executor.execute(plan_result.plan)
        executed = True
        exec_ok = exec_result.success
        log.executed = True
        log.execution_success = exec_result.success
        log.execution_notes = exec_result.notes
        log.final_poses = {oid: [p.x, p.y, p.z] for oid, p in exec_result.final_poses.items()}
        # 6. world-state update
        update_graph_from_execution(graph, exec_result)

    # outcome
    log.success = bool(executed and exec_ok)
    log.failure_category = _failure_category(
        plan_result.valid, plan_result.used_fallback, validation.ok, executed, exec_ok)

    if logger is not None:
        logger.write(log)
    return log, graph

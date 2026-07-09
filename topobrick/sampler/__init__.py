"""Agentic Brick-KG covariate sampler (`pull_v1`).

Given a forecast target sensor in a building's Brick knowledge graph, a picker
LLM selects a small, physics-aware set of covariate time-series from the target's
egocentric neighbourhood. An independent verifier LLM audits that selection. The
sampler reads the knowledge graph only — it never inspects time-series values, so
it carries no information from the forecaster's train split.

Layout, top-down:

    runner.py          CLI. Runs the pipeline over a building's forecast targets.
    build_skeleton.py  CLI. Offline, once per building: KG parquet -> skeleton.json.
    pipeline.py        The orchestrator. One target: render context -> picker LLM
                       -> parse pulls -> resolve to leaves -> optional verifier
                       audit -> subgraph record.
    prompts.py         The two frozen system prompts. Paper artifact; do not edit.

    graph/             The building as a Brick-typed structural graph.
      brick.py           Brick class -> top-type (Point / Equipment / Location /
                         Collection / External), with an override table for
                         classes the shipped ontology cannot resolve.
      skeleton.py        Loads skeleton.json; ascend (ancestry), descend (subtree).
      resolver.py        Structural nodes <-> timeseries leaf points; brick_class
                         lookup; synthetic clusters for skeleton-orphan leaves.

    view/              What the agent is shown, and how its answer is resolved.
      spine.py           The target's ancestry up-cone: its "you are here".
      context.py         Renders the egocentric view (anchors A1..An, pullable
                         class histograms, building-global drivers), and resolves
                         the agent's pull requests back into concrete leaf URIs.

`graph/` and `view/` are libraries; `pipeline` composes them; `runner` is the
entry point. Nothing below `pipeline` calls an LLM.

Quick start:

    python -m topobrick.sampler.build_skeleton --dataset LBNL59
    python -m topobrick.sampler.runner --dataset LBNL59 \\
        --workers 10 --use_verifier --out outputs/subgraphs/LBNL59.json

The picker is stochastic even at temperature=0 (vLLM batching is not bit-stable),
so a re-run will not reproduce a previous subgraph JSON exactly. See
`docs/PROVENANCE.md` for the measured run-to-run agreement and for the exact
two-pass procedure that produced the published artifact.
"""

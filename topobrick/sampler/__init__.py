"""The agentic Brick-KG covariate sampler (`pull_v1`).

For a forecast target, a picker LLM reads the target's egocentric neighbourhood in
the building's knowledge graph and selects a small, physics-aware set of covariate
series; a second LLM audits that selection. The sampler sees the graph only, never
a timeseries value, so it cannot leak anything from the forecaster's train split.

`graph/` and `view/` are libraries, `pipeline` composes them, `runner` is the entry
point. Nothing below `pipeline` calls an LLM.

The picker is stochastic even at temperature 0 (vLLM's batching is not bit-stable),
so a re-run will not reproduce a previous subgraph JSON exactly. `docs/PROVENANCE.md`
has the measured run-to-run agreement, and the two-pass procedure that produced the
published artifact.
"""

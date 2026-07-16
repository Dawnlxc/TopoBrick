from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional

from topobrick.sampler.graph.skeleton import Skeleton
from topobrick.sampler.graph.resolver import LeafResolver
from topobrick.sampler.view.spine import derive_spine
import re

from collections import Counter

from topobrick.sampler.view.context import render_context, resolve_pulls, _short
from topobrick.sampler.prompts import SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT

DEFAULT_MAX_LEAVES = 60


def _parse_pulls_block(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        s = content.rfind("{")
        blob = content[s:] if s >= 0 else None
    for cand in ([blob, blob[:blob.rfind("}") + 1]] if blob else []):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return {}


def call_llm(client, model: str, user_msg: str, max_tokens: int = 16000,
             return_raw: bool = False, system: str = SYSTEM_PROMPT):
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_msg}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    if "gpt-oss" in str(model).lower():
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    parsed = _parse_pulls_block(content)
    return (parsed, content) if return_raw else parsed


def assemble_record(target: str, skel: Skeleton, resolver: LeafResolver,
                    spine: dict, leaves: List[str], llm_out: Dict[str, Any],
                    pull_recs: List[dict], fallback_used: bool) -> Dict[str, Any]:
    anc = [] if spine.get("orphan") else list(spine.get("ancestry_nodes", []))
    anc_edges = [] if spine.get("orphan") else list(spine.get("ancestry_edges", []))
    nodes = set(anc) | {target} | set(leaves)
    edges: List[List[str]] = [list(e) for e in anc_edges if e[0] in nodes and e[2] in nodes]
    for lf in leaves:
        for p in resolver.attach_nodes(lf):
            if p in nodes:
                edges.append([p, "hasPoint", lf]); break
    return {
        "target_uri": target,
        "target_class": resolver.brick_class.get(target, ""),
        "nodes": sorted(nodes),
        "edges": edges,
        "n_leaves": len(leaves),
        "leaves": leaves,
        "spine_size": len(anc),
        "orphan": bool(spine.get("orphan")),
        "roots_reached": spine.get("roots_reached", []),
        "llm_used": not fallback_used,
        "fallback_used": fallback_used,
        "target_interpretation": llm_out.get("target_interpretation", ""),
        "target_structural_summary": llm_out.get("target_structural_summary", ""),
        "pulls": pull_recs,
        "llm_summary": llm_out.get("summary", ""),
    }


def _retry_llm(client, model, user_msg, system):
    last_err = None
    for delay in [0, 5, 15]:
        if delay:
            time.sleep(delay)
        try:
            return call_llm(client, model, user_msg, system=system), None
        except Exception as e:
            last_err = e
            if not any(k in str(e).lower() for k in
                       ("429", "rate", "timeout", "connection", "overloaded",
                        "decode", "json")):
                break
    return {}, last_err


def run_verifier(client, model, ctx_text, leaves, anchor_map,
                 target_uri, skel, resolver):
    """Independent audit: sees the same neighborhood + the picker's chosen set,
    returns (added_leaves, drop_classes, verdict_record)."""
    picked = Counter(resolver.brick_class.get(l, "?") for l in leaves)
    summary = ", ".join(f"{c}×{n}" for c, n in picked.most_common())
    vmsg = (ctx_text + "\n\nCOVARIATES THE ENGINEER CHOSE (audit these):\n  "
            + (summary or "(none)"))
    vout, _ = _retry_llm(client, model, vmsg, VERIFIER_SYSTEM_PROMPT)
    add_leaves, _ = resolve_pulls(vout.get("add", []), anchor_map,
                                  target_uri, skel, resolver)
    drop = set(vout.get("drop", []) or [])
    return add_leaves, drop, {
        "verdict": vout.get("verdict", ""),
        "added": len(add_leaves),
        "dropped_classes": sorted(drop),
        "note": vout.get("note", ""),
    }


def process_one_target(target_uri: str, skel: Skeleton, resolver: LeafResolver,
                       client, model: str, *,
                       max_leaves: int = DEFAULT_MAX_LEAVES,
                       use_verifier: bool = False) -> Dict[str, Any]:
    spine = derive_spine(target_uri, skel, resolver)
    ctx_text, anchor_map = render_context(target_uri, skel, spine, resolver)

    llm_out, last_err = _retry_llm(client, model, ctx_text, SYSTEM_PROMPT)
    pulls = llm_out.get("pulls", []) if last_err is None else []
    leaves, pull_recs = resolve_pulls(pulls, anchor_map, target_uri, skel, resolver)

    verifier_rec = None
    if use_verifier and last_err is None:
        add_leaves, drop, verifier_rec = run_verifier(
            client, model, ctx_text, leaves, anchor_map,
            target_uri, skel, resolver)
        kept = [l for l in leaves if resolver.brick_class.get(l, "?") not in drop]
        seen = set(kept)
        for l in add_leaves:
            if l not in seen:
                seen.add(l)
                kept.append(l)
        leaves = kept

    n_pre = len(leaves)
    leaves = leaves[:max_leaves]

    rec = assemble_record(target_uri, skel, resolver, spine, leaves,
                          llm_out, pull_recs, fallback_used=(last_err is not None))
    rec["n_pre_cap"] = n_pre
    if verifier_rec is not None:
        rec["verifier"] = verifier_rec
    if last_err is not None:
        rec["llm_error"] = str(last_err)
    return rec

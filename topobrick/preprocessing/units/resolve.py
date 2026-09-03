"""Reconcile the declared unit with the inferred one, per sensor."""

from __future__ import annotations

import os

import pandas as pd

from topobrick.utils.progress import log


def resolve_units(processed_directory: str) -> pd.DataFrame:
    """Merge declared and inferred units into `kg_nodes.resolved_unit`."""
    inferred = (
        pd.read_csv(os.path.join(processed_directory, "inferred_units.csv"))
        if os.path.exists(os.path.join(processed_directory, "inferred_units.csv"))
        else pd.DataFrame(columns=["sensor_id", "inferred_unit"])
    )
    source = (
        pd.read_csv(os.path.join(processed_directory, "source_units.csv"))
        if os.path.exists(os.path.join(processed_directory, "source_units.csv"))
        else pd.DataFrame(columns=["sensor_id", "source_unit"])
    )

    # Merge
    s = source[["sensor_id", "source_unit"]].rename(columns={"source_unit": "src"})
    i = inferred[["sensor_id", "inferred_unit"]].rename(
        columns={"inferred_unit": "inf"}
    )
    df = i.merge(s, on="sensor_id", how="outer")

    def _resolve(r):
        sx, ix = r.get("src"), r.get("inf")
        sx = None if pd.isna(sx) else str(sx)
        ix = None if pd.isna(ix) else str(ix)
        if sx and ix:
            if sx == ix:
                return sx, False, "agree"
            return ix, True, "disagree:trust_inferred"
        if sx:
            return sx, False, "source_only"
        if ix:
            return ix, False, "inferred_only"
        return None, False, "unknown"

    res = df.apply(_resolve, axis=1, result_type="expand")
    res.columns = ["resolved_unit", "unit_conflict_with_source", "provenance"]
    df = pd.concat([df, res], axis=1)
    df = df.rename(columns={"src": "source_unit", "inf": "inferred_unit"})
    out_csv = os.path.join(processed_directory, "units_resolved.csv")
    df.to_csv(out_csv, index=False)
    log(f"wrote {out_csv}")

    # Also write into kg_nodes.parquet via point_map.kg_node_id
    pmap = pd.read_parquet(os.path.join(processed_directory, "point_map.parquet"))
    kg_nodes_path = os.path.join(processed_directory, "kg_nodes.parquet")
    nodes = pd.read_parquet(kg_nodes_path)
    sid2nid = dict(zip(pmap["sensor_id"].astype(str), pmap["kg_node_id"].astype(str)))
    df["kg_node_id"] = df["sensor_id"].astype(str).map(sid2nid)
    keep = df.dropna(subset=["kg_node_id"]).drop_duplicates(
        subset=["kg_node_id"], keep="first"
    )
    nid2 = keep.set_index("kg_node_id")
    for col in (
        "inferred_unit",
        "source_unit",
        "resolved_unit",
        "unit_conflict_with_source",
    ):
        nodes[col] = nodes["node_id"].map(nid2[col])
    nodes.to_parquet(kg_nodes_path, index=False)
    log(f"merged unit columns into {kg_nodes_path}")

    # Console summary
    print()
    print(f"=== Resolved units summary: {processed_directory} ===")
    print(f"  total sensors:     {len(df)}")
    resolved_count = df["resolved_unit"].notna().sum()
    resolved_fraction = df["resolved_unit"].notna().mean()
    print(
        f"  resolved:          {resolved_count} ({resolved_fraction:.1%})"
    )
    print(f"  conflict (src!=inf): {df['unit_conflict_with_source'].sum()}")
    print()
    print("provenance breakdown:")
    print(df["provenance"].value_counts().to_string())
    print()
    print("resolved unit distribution:")
    print(df["resolved_unit"].fillna("UNKNOWN").value_counts().head(15).to_string())
    return df

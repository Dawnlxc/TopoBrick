"""Resolve a Brick class name to a top-type -- one of
{Point, Equipment, Location, Collection, External}.

Support module for the offline skeleton build (build_skeleton): the top-type
partitions a building's KG nodes into structure vs. timeseries leaves, which is
what build_skeleton uses to decide what goes into skeleton.json.
  - structural node <- top-type in STRUCTURAL_TYPES (Equipment/Location/Collection/External)
  - pullable leaf   <- top-type == "Point"  (intersected with has_ts downstream)

Most classes resolve through the Brick ontology's subClassOf* hierarchy
(317/346 seen across LBNL59 + BTS_Site_{A,B,C}); a manual override table covers
the rest: typos in class names (e.g. Locationoratory -> Location), BTS-vs-LBNL
naming (Outdoor_* here vs Outside_* in Brick 1.2.1) and classes newer than the
shipped ontology, and dataset extensions that are not Brick classes
(Unknown_Ext, Sensor_Ext, Electrical_Circuit).

`brick:System` is a subclass of `brick:Collection` (Brick 1.2.1), so System
nodes resolve to "Collection" and ARE structural.
"""
from __future__ import annotations
import json
import os
from typing import Dict, Iterable, List, Optional

BRICK_NS = "https://brickschema.org/schema/Brick#"

# The Brick ontology TTL, bundled at topobrick/data/. Resolved relative to the
# package, not the working directory. Only the offline skeleton build reads it;
# the runtime sampler never does. Override with $TOPOBRICK_BRICK_TTL.
DEFAULT_TTL = os.environ.get(
    "TOPOBRICK_BRICK_TTL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "Brick1.2.1.ttl"))

# Top-level Brick classes we resolve against (priority order).
TOP_TYPES = ("Point", "Equipment", "Location", "Collection")

# "External" is synthetic (the dataset's site/weather host `Unknown_Ext`, which
# hosts OAT/Solar on some BTS buildings); treated as structural so its leaves
# stay reachable.
STRUCTURAL_TYPES = frozenset({"Equipment", "Location", "Collection", "External"})

# Classes the shipped Brick ontology cannot resolve. Verified: these 29 are the
# complete unresolved set across LBNL59 + BTS_Site_{A,B,C}.
MANUAL_OVERRIDES: Dict[str, str] = {
    # --- Point (sensors / setpoints / meters / flow / speed / count) ---
    "Chilled_Water_Supply_Flow_Rate": "Point",
    "Hot_Water_Supply_Flow_Rate": "Point",
    "Outdoor_Air_Flow_Rate": "Point",
    "Outdoor_Air_Temperature_Sensor": "Point",   # Brick 1.2.1 uses Outside_*
    "Outdoor_Air_Humidity_Sensor": "Point",
    "Hot_Water_Temperature_Sensor": "Point",
    "Electrical_Energy_Sensor": "Point",
    "Electrical_Generation_Meter": "Point",
    "Electrical_Storage_Meter": "Point",
    "Economizer_Setpoint": "Point",
    "Return_Air_Fan_Speed": "Point",
    "Supply_Air_Fan_Speed": "Point",
    "Occupant_Count": "Point",
    "Sensor_Ext": "Point",
    # --- Equipment ---
    "Electrical_Circuit": "Equipment",
    "Outdoor_Air_Damper": "Equipment",
    "Solar_Inverter": "Equipment",
    "Wifi_AP": "Equipment",
    # --- Location (mostly preprocessing typos) ---
    "Locationoratory": "Location",   # typo of Laboratory
    "Kitchen": "Location",
    "_Kitchen": "Location",
    "Room_Kitchen": "Location",
    "space_Kitchen": "Location",
    "space": "Location",
    "Open_": "Location",
    "Open_space": "Location",
    "Enclosed_space": "Location",
    "Shared_space": "Location",
    # --- External (synthetic site/weather host; not a Brick class) ---
    "Unknown_Ext": "External",
}

# Type assigned when neither the ontology nor the override table resolves a
# class. Conservative: keep it OUT of the skeleton (treated as a leaf candidate
# via has_ts), and surface it loudly so an override can be added.
FALLBACK_TYPE = "Point"


def _build_ancestor_resolver(ttl_path: str):
    """Load the Brick ontology once; return a fn class_name -> top_type|None."""
    from rdflib import Graph, RDFS, URIRef
    g = Graph()
    g.parse(ttl_path, format="turtle")
    top_uris = {t: URIRef(BRICK_NS + t) for t in TOP_TYPES}

    def _ancestors(curi):
        seen, stack = set(), [curi]
        while stack:
            x = stack.pop()
            for sup in g.objects(x, RDFS.subClassOf):
                if sup not in seen:
                    seen.add(sup)
                    stack.append(sup)
        return seen

    def resolve(name: str) -> Optional[str]:
        curi = URIRef(BRICK_NS + name)
        anc = _ancestors(curi) | {curi}
        for t in TOP_TYPES:
            if top_uris[t] in anc:
                return t
        return None

    return resolve


def resolve_types(classes: Iterable[str], ttl_path: str = DEFAULT_TTL,
                  cache_path: Optional[str] = None,
                  verbose: bool = True) -> Dict[str, str]:
    """Map each brick_class to a Brick top-type.

    Overrides win first; the remainder go through the ontology. Anything still
    unresolved gets FALLBACK_TYPE and is reported.
    """
    classes = sorted(set(c for c in classes if c))

    cache: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    out: Dict[str, str] = {}
    need_ontology: List[str] = []
    for c in classes:
        if c in MANUAL_OVERRIDES:
            out[c] = MANUAL_OVERRIDES[c]
        elif c in cache:
            out[c] = cache[c]
        else:
            need_ontology.append(c)

    if need_ontology:
        resolve = _build_ancestor_resolver(ttl_path)
        unresolved: List[str] = []
        for c in need_ontology:
            t = resolve(c)
            if t is None:
                t = FALLBACK_TYPE
                unresolved.append(c)
            out[c] = t
        if unresolved and verbose:
            print(f"[brick_type] WARNING: {len(unresolved)} classes unresolved "
                  f"by ontology+overrides, defaulted to {FALLBACK_TYPE!r}:")
            for c in unresolved:
                print(f"             - {c}")

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(out, f, indent=0, sort_keys=True)

    return out


def is_structural(brick_type: str) -> bool:
    return brick_type in STRUCTURAL_TYPES

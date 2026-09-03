from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Callable

from rdflib import RDFS, Graph, URIRef

BRICK_NAMESPACE = "https://brickschema.org/schema/Brick#"
DEFAULT_ONTOLOGY_PATH = os.environ.get(
    "TOPOBRICK_BRICK_TTL",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "resources",
        "Brick1.2.1.ttl",
    ),
)
TOP_LEVEL_TYPES = ("Point", "Equipment", "Location", "Collection")
STRUCTURAL_TYPES = frozenset({"Equipment", "Location", "Collection", "External"})
FALLBACK_TYPE = "Point"

CLASS_TYPE_OVERRIDES: dict[str, str] = {
    "Chilled_Water_Supply_Flow_Rate": "Point",
    "Hot_Water_Supply_Flow_Rate": "Point",
    "Outdoor_Air_Flow_Rate": "Point",
    "Outdoor_Air_Temperature_Sensor": "Point",
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
    "Electrical_Circuit": "Equipment",
    "Outdoor_Air_Damper": "Equipment",
    "Solar_Inverter": "Equipment",
    "Wifi_AP": "Equipment",
    "Locationoratory": "Location",
    "Kitchen": "Location",
    "_Kitchen": "Location",
    "Room_Kitchen": "Location",
    "space_Kitchen": "Location",
    "space": "Location",
    "Open_": "Location",
    "Open_space": "Location",
    "Enclosed_space": "Location",
    "Shared_space": "Location",
    "Unknown_Ext": "External",
}


def _ontology_resolver(ontology_path: str) -> Callable[[str], str | None]:
    graph = Graph()
    graph.parse(ontology_path, format="turtle")
    top_level_uris = {
        name: URIRef(BRICK_NAMESPACE + name) for name in TOP_LEVEL_TYPES
    }

    def resolve(class_name: str) -> str | None:
        class_uri = URIRef(BRICK_NAMESPACE + class_name)
        ancestors = {class_uri}
        pending = [class_uri]
        while pending:
            current = pending.pop()
            for parent in graph.objects(current, RDFS.subClassOf):
                if parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        return next(
            (name for name, uri in top_level_uris.items() if uri in ancestors),
            None,
        )

    return resolve


def resolve_top_level_types(
    classes: Iterable[str],
    ontology_path: str = DEFAULT_ONTOLOGY_PATH,
    cache_path: str | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    class_names = sorted({name for name in classes if name})
    cache: dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as handle:
            cache = json.load(handle)

    resolved: dict[str, str] = {}
    unresolved = []
    pending = []
    for class_name in class_names:
        if class_name in CLASS_TYPE_OVERRIDES:
            resolved[class_name] = CLASS_TYPE_OVERRIDES[class_name]
        elif class_name in cache:
            resolved[class_name] = cache[class_name]
        else:
            pending.append(class_name)

    if pending:
        resolve = _ontology_resolver(ontology_path)
        for class_name in pending:
            top_level_type = resolve(class_name)
            if top_level_type is None:
                top_level_type = FALLBACK_TYPE
                unresolved.append(class_name)
            resolved[class_name] = top_level_type

    if unresolved and verbose:
        print(
            f"[brick] {len(unresolved)} unresolved classes defaulted to "
            f"{FALLBACK_TYPE}: {', '.join(unresolved)}"
        )

    if cache_path:
        with open(cache_path, "w") as handle:
            json.dump(resolved, handle, sort_keys=True)
    return resolved


def is_structural_type(top_level_type: str) -> bool:
    return top_level_type in STRUCTURAL_TYPES

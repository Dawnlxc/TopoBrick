"""The building as a Brick-typed structural graph.

    brick      Brick class -> top-type (Point / Equipment / Location / ...).
    skeleton   Loads skeleton.json; ascend (ancestry) and descend (subtree).
    resolver   Maps structural nodes <-> timeseries leaf points.

Nothing here touches time-series values. `build_skeleton` (one level up) writes
the skeleton.json that `skeleton` reads.
"""

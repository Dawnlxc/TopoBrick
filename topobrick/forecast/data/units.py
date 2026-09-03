"""Canonical unit conversion for time-series values."""

from __future__ import annotations

from typing import Optional

_ALIAS = {
    "°C": "degC",
    "degC": "degC",
    "DegreeCelsius": "degC",
    "C": "degC",
    "°F": "degF",
    "degF": "degF",
    "DegreeFahrenheit": "degF",
    "F": "degF",
    "K": "K",
    "%": "%",
    "Percent": "%",
    "percent": "%",
    "ppm": "ppm",
    "psi": "psi",
    "Pa": "Pa",
    "kPa": "kPa",
    "InchesH2O": "inH2O",
    "cfm": "cfm",
    "CFM": "cfm",
    "L/s": "L/s",
    "m3/s": "m3/s",
    "kW": "kW",
    "KiloW": "kW",
    "Kilowatt": "kW",
    "kW (or W)": "kW",
    "W": "W",
    "Watt": "W",
    "kVA": "kVA",
    "KiloV-A": "kVA",
    "kVAR": "kVAR",
    "KiloV-A_Reactive": "kVAR",
    "KiloV-A-Reactive": "kVAR",
    "kWh": "kWh",
    "KiloW-HR": "kWh",
    "Kilowatt-Hour": "kWh",
    "Wh": "Wh",
    "kVAR-HR": "kVARh",
    "KiloV-A_Reactive-HR": "kVARh",
    "V-A_Reactive-HR": "VARh",
    "MBTU/h": "MBTU/h",
    "mbtuph": "MBTU/h",
    "V": "V",
    "Volt": "V",
    "A": "A",
    "Amp": "A",
    "Ampere": "A",
    "Hz": "Hz",
    "Hertz": "Hz",
    "W/m^2": "W/m^2",
    "W/m2": "W/m^2",
    "DEG": "deg",
    "/": None,
    "": None,
    "None": None,
    "UNKNOWN": None,
}


def canonicalise_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    s = str(u).strip()
    if s in _ALIAS:
        return _ALIAS[s]
    for k, v in _ALIAS.items():
        if k.lower() == s.lower():
            return v
    return s


CANONICAL_TEMPERATURE = "degC"


def temperature_to_canonical(values, unit: Optional[str]):
    u = canonicalise_unit(unit)
    if u is None or u == CANONICAL_TEMPERATURE:
        return values
    if u == "degF":
        return (values - 32.0) * (5.0 / 9.0)
    if u == "K":
        return values - 273.15
    return values

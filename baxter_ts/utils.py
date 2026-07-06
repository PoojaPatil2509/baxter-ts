"""
Shared helpers for baxter-ts.

normalize_freq() exists because pandas has grown several generations of
frequency aliases ("T" vs "min", "H" vs "h", "M" vs "ME", "A" vs "YS"...)
and different parts of the pipeline were keyed on different generations.
v0.1.x bug: DatetimeValidator emitted "min"/"h" while the feature engineer
looked up "T"/"H" — so hourly and minutely data silently received daily
lag defaults. Every component now normalizes through this one function.
"""

from typing import Optional

# Canonical frequency tokens used across the pipeline
CANONICAL_FREQS = ("min", "h", "D", "W", "MS", "Q", "YS")

_ALIAS_MAP = {
    # sub-daily
    "s": "min", "S": "min", "ms": "min", "L": "min",
    "t": "min", "T": "min", "min": "min",
    "h": "h", "H": "h", "bh": "h", "BH": "h",
    # daily
    "d": "D", "D": "D", "b": "D", "B": "D", "c": "D", "C": "D",
    # weekly
    "w": "W", "W": "W",
    # monthly (lowercase "ms" is milliseconds, handled above — pandas
    # emits month-start strictly as uppercase "MS")
    "m": "MS", "M": "MS", "me": "MS", "ME": "MS",
    "MS": "MS", "sm": "MS", "SM": "MS",
    "sms": "MS", "SMS": "MS",
    # quarterly
    "q": "Q", "Q": "Q", "qs": "Q", "QS": "Q", "qe": "Q", "QE": "Q",
    "bq": "Q", "BQ": "Q",
    # yearly
    "a": "YS", "A": "YS", "y": "YS", "Y": "YS",
    "ys": "YS", "YS": "YS", "ye": "YS", "YE": "YS",
    "as": "YS", "AS": "YS", "ba": "YS", "BA": "YS",
}


def normalize_freq(freq: Optional[str], default: str = "D") -> str:
    """
    Map any pandas frequency alias to one of CANONICAL_FREQS.

    Handles anchored offsets ("W-SUN", "Q-DEC", "A-JAN") and multiples
    ("15min", "2H") by reducing to the base unit.
    """
    if not freq:
        return default
    base = str(freq).split("-")[0].strip()
    # strip a leading multiplier: "15min" -> "min", "2H" -> "H"
    base = base.lstrip("0123456789 ")
    if not base:
        return default
    if base in _ALIAS_MAP:
        return _ALIAS_MAP[base]
    # Case-insensitive fallback ("Min", "Ms" from odd sources)
    return _ALIAS_MAP.get(base.lower(), _ALIAS_MAP.get(base.upper(), default))

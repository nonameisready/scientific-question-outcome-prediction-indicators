"""Lexicon of major astronomical facilities with operational epochs.

Used for (a) tractability features — whether a question names an instrument
that already existed at the cutoff, (b) field-environment features — proximity
of a subfield's next major facility launch to the cutoff, and (c) evidence
features — counting distinct instruments in retrieved evidence.

`start_year` is the year of first science operations (for GW detectors, the
first observing run of the advanced configuration). `None` means the facility
was not yet operational at the end of the study period (2025).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Facility:
    key: str
    pattern: str
    start_year: int | None


FACILITIES: dict[str, Facility] = {
    f.key: f
    for f in (
        Facility("hst", r"\b(?:hst|hubble space telescope)\b", 1990),
        Facility("vlt", r"\b(?:vlt|very large telescope)\b", 1998),
        Facility("chandra", r"\bchandra\b", 1999),
        Facility("xmm", r"\bxmm(?:-newton)?\b", 1999),
        Facility("sdss", r"\b(?:sdss|sloan digital sky survey)\b", 2000),
        Facility("wmap", r"\bwmap\b", 2001),
        Facility("spitzer", r"\bspitzer\b", 2003),
        Facility("harps", r"\bharps\b", 2003),
        Facility("swift", r"\bswift\b", 2004),
        Facility("fermi", r"\bfermi(?:-lat| lat)?\b", 2008),
        Facility("kepler", r"\bkepler\b", 2009),
        Facility("herschel", r"\bherschel\b", 2009),
        Facility("planck", r"\bplanck\b", 2009),
        Facility("wise", r"\b(?:wise|neowise)\b", 2009),
        Facility("icecube", r"\bicecube\b", 2010),
        Facility("alma", r"\b(?:alma|atacama large millimeter)\b", 2011),
        Facility("nustar", r"\bnustar\b", 2012),
        Facility("askap", r"\baskap\b", 2012),
        Facility("lofar", r"\blofar\b", 2012),
        Facility("gaia", r"\bgaia\b", 2013),
        Facility("sphere", r"\bsphere\b", 2014),
        Facility("gpi", r"\b(?:gpi|gemini planet imager)\b", 2014),
        Facility("ligo", r"\b(?:ligo|advanced ligo|aligo)\b", 2015),
        Facility("fast_telescope", r"\bfive[- ]hundred[- ]meter aperture|\bfast\b.{0,30}radio telescope", 2016),
        Facility("virgo", r"\b(?:advanced )?virgo\b", 2017),
        Facility("nicer", r"\bnicer\b", 2017),
        Facility("eht", r"\b(?:eht|event horizon telescope)\b", 2017),
        Facility("ztf", r"\b(?:ztf|zwicky transient facility)\b", 2018),
        Facility("tess", r"\btess\b", 2018),
        Facility("chime", r"\bchime\b", 2018),
        Facility("meerkat", r"\bmeerkat\b", 2018),
        Facility("espresso", r"\bespresso\b", 2018),
        Facility("desi", r"\b(?:desi|dark energy spectroscopic instrument)\b", 2019),
        Facility("erosita", r"\berosita\b", 2019),
        Facility("kagra", r"\bkagra\b", 2020),
        Facility("nanograv", r"\b(?:nanograv|pulsar timing array|epta|ppta)\b", 2013),
        Facility("jwst", r"\b(?:jwst|james webb space telescope|webb telescope)\b", 2022),
        Facility("euclid", r"\beuclid\b", 2023),
        Facility("rubin", r"\b(?:rubin observatory|lsst|legacy survey of space and time)\b", 2025),
        Facility("roman", r"\b(?:roman space telescope|wfirst)\b", None),
        Facility("elt", r"\b(?:elt|extremely large telescope|tmt|thirty meter telescope|gmt|giant magellan)\b", None),
        Facility("ska", r"\b(?:ska|square kilomet(?:er|re) array)\b", None),
        Facility("lisa", r"\blisa\b", None),
        Facility("cta", r"\b(?:cta|cherenkov telescope array)\b", None),
        Facility("ariel", r"\bariel\b", None),
        Facility("parkes", r"\bparkes\b", 1961),
        Facility("arecibo", r"\barecibo\b", 1963),
        Facility("vla", r"\b(?:vla|very large array|jvla)\b", 1980),
        Facility("keck", r"\bkeck\b", 1993),
        Facility("gbt", r"\b(?:gbt|green bank telescope)\b", 2000),
    )
}

_COMPILED = {key: re.compile(f.pattern, re.IGNORECASE) for key, f in FACILITIES.items()}


def find_facilities(text: str) -> list[str]:
    """Return facility keys mentioned in the text."""
    return [key for key, rx in _COMPILED.items() if rx.search(text)]


def operational_at(facility_key: str, year: int) -> bool:
    start = FACILITIES[facility_key].start_year
    return start is not None and start <= year


def next_launch_gap_years(facility_keys: tuple[str, ...], cutoff_year: int) -> float | None:
    """Years from cutoff until the next relevant facility becomes operational.

    Returns 0 if a relevant facility launched in the cutoff year, the smallest
    positive gap otherwise, and None when every relevant facility is already
    operational (or has no known start)."""
    gaps = []
    for key in facility_keys:
        start = FACILITIES[key].start_year
        if start is not None and start >= cutoff_year:
            gaps.append(start - cutoff_year)
    return min(gaps) if gaps else None

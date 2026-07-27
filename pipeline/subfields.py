"""Astronomy subfield definitions and keyword-based paper classification.

Eight subfields spanning distinct observational and theoretical communities.
A paper is assigned to every subfield whose keyword score passes the
threshold, because real papers frequently straddle communities; per-subfield
corpora may therefore overlap slightly. Classification uses only title and
abstract text, so it is identical at every historical cutoff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Subfield:
    key: str
    name: str
    # (pattern, weight) pairs matched case-insensitively on title+abstract.
    patterns: tuple[tuple[str, int], ...]
    # arXiv categories that make membership more plausible (soft prior).
    preferred_categories: tuple[str, ...] = field(default=())
    # Major facilities relevant to the subfield (keys into facilities.FACILITIES).
    facilities: tuple[str, ...] = field(default=())


SUBFIELDS: tuple[Subfield, ...] = (
    Subfield(
        key="exoplanet_atmospheres",
        name="Exoplanet atmospheres",
        patterns=(
            (r"exoplanet(?:ary)? atmospheres?", 4),
            (r"transmission spectr(?:um|a|oscopy)", 4),
            (r"atmospheric retrievals?", 4),
            (r"secondary eclipse", 3),
            (r"emission spectr\w+ of (?:the )?(?:hot|warm|exo)", 3),
            (r"hot jupiters?", 2),
            (r"atmospheric escape", 3),
            (r"(?:water|h2o|methane|co2) (?:abundance|detection|feature)s?", 2),
            (r"thermal inversion", 3),
            (r"(?:cloud|haze)s? in (?:the )?atmospheres?", 3),
            (r"phase curve", 2),
            (r"habitable[- ]zone", 1),
            (r"biosignatures?", 2),
        ),
        preferred_categories=("astro-ph.EP",),
        facilities=("jwst", "hst", "spitzer", "tess", "kepler", "espresso"),
    ),
    Subfield(
        key="protoplanetary_disks",
        name="Protoplanetary disks and planet formation",
        patterns=(
            (r"protoplanetary dis[kc]s?", 4),
            (r"planet(?:esimal)? formation", 3),
            (r"circumstellar dis[kc]s?", 3),
            (r"transition(?:al)? dis[kc]s?", 3),
            (r"dust (?:rings?|gaps?|traps?|growth)", 3),
            (r"pebble accretion", 4),
            (r"snow ?lines?", 3),
            (r"dis[kc] (?:winds?|chemistry|substructures?)", 3),
            (r"core accretion", 2),
            (r"debris dis[kc]s?", 2),
            (r"t tauri", 2),
            (r"herbig ae", 2),
        ),
        preferred_categories=("astro-ph.EP", "astro-ph.SR"),
        facilities=("alma", "jwst", "sphere", "herschel", "vlt"),
    ),
    Subfield(
        key="stellar_activity",
        name="Stellar magnetic activity and flares",
        patterns=(
            (r"stellar (?:flares?|activity|cycles?)", 4),
            (r"starspots?", 4),
            (r"chromospheric activity", 4),
            (r"superflares?", 4),
            (r"stellar coronae?", 3),
            (r"(?:stellar|m[- ]dwarf) magnetic (?:fields?|activity)", 3),
            (r"activity cycles?", 3),
            (r"stellar (?:rotation|dynamo)", 2),
            (r"coronal mass ejections?", 2),
            (r"flare (?:rates?|energies|frequency)", 3),
            (r"ca ?ii h ?& ?k", 3),
        ),
        preferred_categories=("astro-ph.SR",),
        facilities=("kepler", "tess", "chandra", "xmm", "gaia"),
    ),
    Subfield(
        key="fast_radio_bursts",
        name="Fast radio bursts",
        patterns=(
            (r"fast radio bursts?", 5),
            (r"\bfrbs?\b", 4),
            (r"repeating (?:radio )?bursts?", 3),
            (r"radio bursts? .{0,40}extragalactic", 3),
            (r"rotating radio transients?", 2),
            (r"lorimer burst", 4),
            (r"perytons?", 2),
        ),
        preferred_categories=("astro-ph.HE",),
        facilities=("chime", "askap", "parkes", "fast_telescope", "meerkat"),
    ),
    Subfield(
        key="gravitational_waves",
        name="Gravitational-wave astrophysics",
        patterns=(
            (r"gravitational[- ]waves?", 4),
            (r"\bligo\b", 3),
            (r"binary black hole(?:s| merger)", 3),
            (r"neutron star mergers?", 3),
            (r"compact binary coalescences?", 4),
            (r"stochastic (?:gravitational[- ]wave )?background", 3),
            (r"pulsar timing array", 3),
            (r"kilonovae?", 3),
            (r"gw\d{6}", 4),
            (r"chirp mass", 3),
            (r"standard sirens?", 4),
        ),
        preferred_categories=("astro-ph.HE", "astro-ph.CO"),
        facilities=("ligo", "virgo", "kagra", "lisa", "nanograv"),
    ),
    Subfield(
        key="galaxy_evolution",
        name="Galaxy formation and evolution",
        patterns=(
            (r"galaxy (?:formation|evolution|quenching)", 4),
            (r"star[- ]formation (?:histor(?:y|ies)|rates?|quenching)", 3),
            (r"stellar mass functions?", 3),
            (r"high[- ]redshift galax(?:y|ies)", 3),
            (r"quiescent galax(?:y|ies)", 3),
            (r"galaxy mergers?", 2),
            (r"(?:agn|stellar) feedback", 3),
            (r"main sequence of star[- ]forming", 3),
            (r"mass[- ]metallicity relation", 3),
            (r"reionization", 2),
            (r"lyman[- ]break", 2),
            (r"galactic outflows?", 2),
        ),
        preferred_categories=("astro-ph.GA",),
        facilities=("jwst", "hst", "alma", "sdss", "euclid"),
    ),
    Subfield(
        key="cosmology_tensions",
        name="Cosmological parameters and tensions",
        patterns=(
            (r"hubble (?:constant|tension|parameter)", 4),
            (r"\bh_?0\b.{0,60}(?:measurement|tension|determination)", 4),
            (r"(?:sigma|s)_?8 tension", 4),
            (r"dark energy", 3),
            (r"cosmological (?:parameters?|constraints?)", 3),
            (r"cosmic microwave background", 2),
            (r"baryon acoustic oscillations?", 3),
            (r"type ia supernova\w* .{0,40}(?:cosmolog|distance)", 3),
            (r"cosmic distance ladder", 4),
            (r"early dark energy", 4),
            (r"\blambda ?cdm\b", 2),
            (r"weak (?:gravitational )?lensing .{0,40}(?:survey|cosmolog)", 2),
        ),
        preferred_categories=("astro-ph.CO",),
        facilities=("planck", "desi", "euclid", "rubin", "sdss"),
    ),
    Subfield(
        key="compact_objects",
        name="Compact objects and accretion",
        patterns=(
            (r"neutron stars?", 3),
            (r"magnetars?", 3),
            (r"x[- ]ray binar(?:y|ies)", 3),
            (r"black hole (?:accretion|binar|spin|mass)", 3),
            (r"pulsars?", 2),
            (r"accretion dis[kc]s?", 2),
            (r"equation of state .{0,40}(?:dense|neutron|nuclear)", 4),
            (r"quasi[- ]periodic oscillations?", 3),
            (r"tidal disruption events?", 3),
            (r"ultraluminous x[- ]ray", 3),
            (r"gamma[- ]ray bursts?", 2),
            (r"millisecond pulsars?", 3),
        ),
        preferred_categories=("astro-ph.HE",),
        facilities=("nicer", "chandra", "xmm", "nustar", "eht", "fermi"),
    ),
)

SUBFIELD_KEYS = tuple(s.key for s in SUBFIELDS)

# Papers scoring at or above this on a subfield are assigned to it.
SCORE_THRESHOLD = 4
CATEGORY_BONUS = 1


def _compile(subfield: Subfield) -> list[tuple[re.Pattern, int]]:
    return [(re.compile(p, re.IGNORECASE), w) for p, w in subfield.patterns]


_COMPILED = {s.key: _compile(s) for s in SUBFIELDS}
_BY_KEY = {s.key: s for s in SUBFIELDS}


def get_subfield(key: str) -> Subfield:
    return _BY_KEY[key]


def classify(title: str, abstract: str, categories: list[str]) -> dict[str, int]:
    """Return {subfield_key: score} for subfields passing the threshold."""
    text = f"{title} {abstract}"
    scores: dict[str, int] = {}
    for sub in SUBFIELDS:
        score = 0
        for pattern, weight in _COMPILED[sub.key]:
            hits = len(pattern.findall(text))
            if hits:
                score += weight * min(hits, 3)
        if score and any(c in sub.preferred_categories for c in categories):
            score += CATEGORY_BONUS
        if score >= SCORE_THRESHOLD:
            scores[sub.key] = score
    return scores

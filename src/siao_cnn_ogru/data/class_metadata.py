"""Class metadata and label mapping for NPPAD-based research subsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ClassMetadata:
    """Metadata for a single accident class."""

    code: str
    full_name: str
    source: str  # "original" or "simulated"


ORIGINAL_NPPAD_CLASSES_18: Dict[str, str] = {
    "Normal": "Normal Operation",
    "ATWS": "Anticipated Transient Without Scram",
    "FLB": "Feedwater Line Break",
    "LACP": "Loss of AC Power",
    "LLB": "Letdown Line Break",
    "LOCA": "Loss of Coolant Accident (Hot Leg)",
    "LOCAC": "Loss of Coolant Accident (Cold Leg)",
    "LOF": "Loss of Flow (Locked Rotor)",
    "LR": "Load Rejection",
    "MD": "Moderator Dilution",
    "RI": "Rod Insertion",
    "RW": "Rod Withdrawal",
    "SGATR": "Steam Generator A Tube Rupture",
    "SGBTR": "Steam Generator B Tube Rupture",
    "SLBIC": "Steam Line Break Inside Containment",
    "SLBOC": "Steam Line Break Outside Containment",
    "SP": "Spark Presence for Hydrogen Burn",
    "TT": "Turbine Trip",
}


RESEARCH_CLASS_CODES_14: Tuple[str, ...] = (
    "Normal",
    "FLB",
    "LLB",
    "LOCA",
    "LOCAC",
    "LR",
    "MD",
    "RI",
    "RW",
    "SGATR",
    "SGBTR",
    "SLBIC",
    "SLBOC",
    "TT",
)


SIMULATED_CLASS_CODES = {"Normal", "TT"}


def resolve_active_class_codes(active_class_codes: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """Resolve and validate active class codes, preserving user-provided order."""
    classes = tuple(active_class_codes) if active_class_codes is not None else RESEARCH_CLASS_CODES_14

    if len(classes) != len(set(classes)):
        raise ValueError(f"Duplicate class codes provided: {classes}")

    unknown = [code for code in classes if code not in ORIGINAL_NPPAD_CLASSES_18]
    if unknown:
        raise ValueError(f"Unknown class codes: {unknown}")

    return classes


def build_label_maps(active_class_codes: Optional[Sequence[str]] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build contiguous label maps for the configured active classes."""
    classes = resolve_active_class_codes(active_class_codes)
    class_to_label = {code: idx for idx, code in enumerate(classes)}
    label_to_class = {idx: code for code, idx in class_to_label.items()}
    return class_to_label, label_to_class


def get_class_metadata(code: str) -> ClassMetadata:
    """Get metadata for a single class code."""
    if code not in ORIGINAL_NPPAD_CLASSES_18:
        raise KeyError(f"Unknown class code: {code}")

    source = "simulated" if code in SIMULATED_CLASS_CODES else "original"
    return ClassMetadata(code=code, full_name=ORIGINAL_NPPAD_CLASSES_18[code], source=source)


def build_label_metadata_map(
    active_class_codes: Optional[Sequence[str]] = None,
) -> Dict[int, ClassMetadata]:
    """Build label-indexed metadata map for active classes."""
    class_to_label, _ = build_label_maps(active_class_codes)
    return {label: get_class_metadata(code) for code, label in class_to_label.items()}


__all__ = [
    "ClassMetadata",
    "ORIGINAL_NPPAD_CLASSES_18",
    "RESEARCH_CLASS_CODES_14",
    "SIMULATED_CLASS_CODES",
    "resolve_active_class_codes",
    "build_label_maps",
    "get_class_metadata",
    "build_label_metadata_map",
]

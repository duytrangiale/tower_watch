"""Damage injection: brace stiffness reduction.

Damage is modeled as EA -> (1 - d) * EA on one or more brace elements
(diagonal or horizontal), matching LUMO's removable-brace damage. Mass is
left unchanged, since the guideline specifies damage as a stiffness
reduction only. See TowerWatch_guideline.md Sec 4.3.
"""

import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class DamageRecord:
    element_idx: int
    severity: float
    element_type: str
    node_i: int
    node_j: int


def brace_element_indices(element_type):
    """Indices of elements eligible for damage: diagonal or horizontal braces, not legs."""
    return np.where((element_type == "diagonal") | (element_type == "horizontal"))[0]


def apply_damage(model, damage_specs):
    """Return a copy of `model` with reduced EA on the given brace elements.

    `damage_specs` is a list of (element_idx, severity) pairs, with severity
    in [0, 1] where 1.0 means fully removed (EA -> 0). Supports single or
    multiple simultaneous brace damage by passing more than one pair.
    """
    damaged_model = copy.deepcopy(model)
    records = []
    for element_idx, severity in damage_specs:
        element_type = model.element_type[element_idx]
        assert element_type in ("diagonal", "horizontal"), (
            f"Damage must target a brace element, got '{element_type}'"
        )
        damaged_model.ea[element_idx] = model.ea[element_idx] * (1.0 - severity)
        i, j = model.elements[element_idx]
        records.append(DamageRecord(
            element_idx=element_idx,
            severity=severity,
            element_type=element_type,
            node_i=int(i),
            node_j=int(j),
        ))
    return damaged_model, records

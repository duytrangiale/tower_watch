"""Temperature confounder: models how ambient temperature shifts a steel
structure's stiffness, and therefore its natural frequencies, by an amount
comparable to real damage. See TowerWatch_guideline.md Sec 5.2.
"""


def modulus_at_temperature(youngs_modulus_pa: float, temperature_c: float,
                            reference_temperature_c: float, coefficient_per_c: float) -> float:
    """E(T) = E0 * (1 - alpha * (T - T0)).

    `coefficient_per_c` (alpha) is steel's approximate thermal sensitivity
    of Young's modulus, on the order of 0.0002-0.0004 per degree C.
    """
    return youngs_modulus_pa * (1.0 - coefficient_per_c * (temperature_c - reference_temperature_c))


def material_at_temperature(material_config: dict, temperature_c: float, environment_config: dict) -> dict:
    """Return a copy of `material_config` with youngs_modulus_pa adjusted
    for `temperature_c`, leaving every other material property unchanged.
    """
    adjusted = dict(material_config)
    adjusted["youngs_modulus_pa"] = modulus_at_temperature(
        material_config["youngs_modulus_pa"],
        temperature_c,
        environment_config["reference_temperature_c"],
        environment_config["modulus_temp_coefficient_per_c"],
    )
    return adjusted

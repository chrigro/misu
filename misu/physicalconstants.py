# -*- coding: utf-8 -*-

"""Constants from scipy.

See https://docs.scipy.org/doc/scipy/reference/constants.html

"""

from . import misulib

try:
    from scipy import constants as const

    have_scipy = True
except ImportError:
    have_scipy = False

available_constants = {
    "Avogadro constant": "na",
    "Bohr magneton": "mu_b",
    "Bohr radius": "a_b",
    "Boltzmann constant": "kb",
    "Planck constant": "h",
    "Planck constant over 2 pi": "hbar",
    "Rydberg constant": "ry",
    "Stefan-Boltzmann constant": "sigma",
    "atomic mass constant": "u",
    "electron g factor": "ge",
    "electron mass": "me",
    "electric constant": "eps0",
    "elementary charge": "e",
    "fine-structure constant": "alpha",
    "mag. constant": "mu0",
    "speed of light in vacuum": "c",
    "standard acceleration of gravity": "g",
}

fallback_values = {
    "Avogadro constant": [6.022140857e+23, "mol^-1"],
    "Bohr magneton": [9.274009994e-24, "J T^-1"],
    "Bohr radius": [5.2917721067e-11, "m"],
    "Boltzmann constant": [1.38064852e-23, "J K^-1"],
    "Planck constant": [6.62607004e-34, "J s"],
    "Planck constant over 2 pi": [1.0545718e-34, "J s"],
    "Rydberg constant": [10973731.568508, "m^-1"],
    "Stefan-Boltzmann constant": [5.670367e-08, "W m^-2 K^-4"],
    "atomic mass constant": [1.66053904e-27, "kg"],
    "electron g factor": [-2.00231930436182, ""],
    "electron mass": [9.10938356e-31, "kg"],
    "electric constant": [8.854187817620389e-12, "F m^-1"],
    "elementary charge": [1.6021766208e-19, "C"],
    "fine-structure constant": [0.0072973525664, ""],
    "mag. constant": [1.2566370614359173e-06, "N A^-2"],
    "speed of light in vacuum": [299792458.0, "m s^-1"],
    "standard acceleration of gravity": [9.80665, "m s^-2"],
}


if have_scipy:
    for name, shortname in available_constants.items():
        val, unit, _ = const.physical_constants[name]
        quant = misulib.quantity_from_string(unit)
        # put them in the module namespace
        globals()[shortname] = val * quant
else:
    for name, shortname in available_constants.items():
        val, unit = fallback_values[name]
        quant = misulib.quantity_from_string(unit)
        # put them in the module namespace
        globals()[shortname] = val * quant


if __name__ == "__main__":
    for cc in const.physical_constants.keys():
        if cc in available_constants.keys():
            print(
                f"{cc} = {const.physical_constants[cc][0]} {const.physical_constants[cc][1]}"
            )

# coding=utf8
from __future__ import division, print_function
import sys
import re
import json
import inspect
import os.path as osp
from misu.siprefixes import siprefixes_sym

import misu.engine as engine


class UnitNamespace(object):
    """A namespace for all defined units.

    """

    def __init__(self, context="default"):
        """Initialize the unit namespace.

        We create all SI base units plus the dimensionless unit.
        See https://physics.nist.gov/cuu/Units/units.html

        Parameters
        ----------
        context : string (default: 'all')
            In which context are we working? Used to restrict the available units.

        """
        # dimensionless is special
        self.dimensionless = engine.Quantity(1.0)
        engine.addType(self.dimensionless, "Dimensionless")
        self.known_units = ["dimensionless"]

        # meter
        self.add_unit(
            symbols=["m", "metre", "metres", "meter", "meters"],
            sidict=dict(m=1.0),
            scale_factor=1.0,
            representative_symbol="m",
            create_metric_prefixes_for=["m"],
            unit_category="Length",
            metric_skip_function=None,
        )

        # kg (special since prefix already included)
        self.add_unit(
            symbols=["g", "grams", "gram"],
            sidict=dict(kg=1.0),
            scale_factor=1.0,
            representative_symbol=None,
            create_metric_prefixes_for=["g"],
            unit_category="Mass",
            metric_skip_function=None,
        )
        self.g.setRepresent(as_unit=self.kg, symbol="kg")

        # seconds
        self.add_unit(
            symbols=["s", "second", "sec", "seconds", "secs"],
            sidict=dict(s=1.0),
            scale_factor=1.0,
            representative_symbol="s",
            create_metric_prefixes_for=["s"],
            unit_category="Time",
            metric_skip_function=lambda p: p in ["a"],  # no "as" since it is a keyword
        )

        # ampere
        self.add_unit(
            symbols=["A", "ampere", "amperes", "amp", "amps"],
            sidict=dict(A=1.0),
            scale_factor=1.0,
            representative_symbol="A",
            create_metric_prefixes_for=["A"],
            unit_category="Electric current",
            metric_skip_function=None,
        )

        # ampere
        self.add_unit(
            symbols=["K", "kelvin"],
            sidict=dict(K=1.0),
            scale_factor=1.0,
            representative_symbol="K",
            create_metric_prefixes_for=["K"],
            unit_category="Thermodynamic temperature",
            metric_skip_function=None,
        )

        # candela
        self.add_unit(
            symbols=["cd", "candela", "ca"],
            sidict=dict(cd=1.0),
            scale_factor=1.0,
            representative_symbol="cd",
            create_metric_prefixes_for=["cd"],
            unit_category="Luminous intensity",
            metric_skip_function=None,
        )

        # mol
        self.add_unit(
            symbols=["mol", "mole", "moles"],
            sidict=dict(mol=1.0),
            scale_factor=1.0,
            representative_symbol="mol",
            create_metric_prefixes_for=["mol"],
            unit_category="Ammount of substance",
            metric_skip_function=None,
        )

        # create derived units according to the context
        self._create_derived_units(context)

    def add_unit(
        self,
        symbols,
        sidict,
        scale_factor,
        representative_symbol,
        create_metric_prefixes_for=[],
        unit_category="",
        metric_skip_function=None,
    ):
        """Add a unit to the namespace.

        Parameters
        ----------
        symbols : list of unit symbols
            These will be put into the class namespace, and will be entered as keys in
            the global UnitRegistry.

        sidict : dict
            Dictionary with base SI units as keys and the exponent as value.

        scale_factor : float
            Scale factor of the quantity.

        representative_symbol : string on None (default: None)
            Symbol that should be used to represent the unit in a result of a calculation.
            Must be in symbols.

        create_metric_prefixes_for : list of unit symbols (default: empty list)
            List of symbols (must also be in symbols) to create derived units with the
            metric prefixes.

        unit_category : string or None (default: None)
            Category the unit belongs to.

        metric_skip_function : callable (default: None)
            Callable that returns true for the metric prefixes for which the prefixed
            unit should not be created.

        """
        # First define the quantity based on si
        quantity = engine.Quantity(scale_factor)
        quantity.setValDict(sidict)
        # Add to category
        if unit_category is not "":
            # print("Adding type {} to category {}".format(quantity.units(), unit_category))
            engine.addType(quantity, unit_category)
        # Representative symbol
        if representative_symbol is not None:
            self._check_represent(representative_symbol, symbols)
            quantity.setRepresent(as_unit=quantity, symbol=representative_symbol)
        # Add to registry and namespace
        for symbol in symbols:
            symbol = symbol.strip()
            if symbol == "":
                continue
            self._add_quant_attr(symbol, quantity)
            self._add_to_registry(symbol, quantity)
        # Metric prefixes for the first symbol
        self._check_metric_prefix_request(create_metric_prefixes_for, symbols)
        for symbol in create_metric_prefixes_for:
            self.create_metric_prefixes(symbol, quantity, metric_skip_function)

    def create_metric_prefixes(self, symbol, quantity, skipfunction=None):
        """ Populates the UnitRegistry and the namespace with all the
        SI-prefixed versions of the given symbol.

        """
        for prefix in siprefixes_sym:
            if skipfunction and skipfunction(prefix):
                continue
            prefsymb = "{p}{s}".format(p=prefix, s=symbol)
            prefquant = 10 ** (float(siprefixes_sym[prefix].exponent)) * quantity
            self._add_quant_attr(prefsymb, prefquant)
            self._add_to_registry(prefsymb, prefquant)

    def _check_metric_prefix_request(self, metricpref_list, symbols):
        """Check if all the symbols we shall create metric prefixes for are in symbols."""
        if not isinstance(metricpref_list, list):
            raise TypeError(
                "The symbols for which we shall create prefixed units must be given as a list."
            )
        for mp in metricpref_list:
            if mp not in symbols:
                raise ValueError(
                    "Can't create metric prefixed units for {s}. Not in symbols {sym}".format(
                        s=mp, sym=symbols
                    )
                )

    def _check_represent(self, symb, symbols):
        """Check if the representing symbol symb is valid."""
        if symb not in symbols:
            raise ValueError(
                "Representative symbol {s} not in symbols {sym}".format(
                    s=symb, sym=symbols
                )
            )

    def _add_quant_attr(self, symb, quant):
        """Add the quantity as an attribute to the namespace."""
        setattr(self, symb, quant)
        self.known_units.append(symb)

    def _add_to_registry(self, symbol, quantity):
        """Add symbol representing Quantity to the UnitRegistry.

        """
        if symbol not in engine.UnitRegistry.keys():
            engine.UnitRegistry[symbol] = quantity
        else:
            raise ValueError(
                "Unit symbol {s} already present in the registry.".format(s=symbol)
            )

    def _create_derived_units(self, context):
        """Create derived units according to the requested context.

        """
        datapath = osp.join(osp.dirname(__file__), "unitdefs.json")
        with open(datapath) as f:
            unitdefs = json.load(f)
        for unitdef in unitdefs.values():
            conts = unitdef["used in contexts"]
            quant = self.quantity_from_string(unitdef["representation in SI or earlier defined unit"])
            if unitdef["skipped prefixes"]:
                skipfcn = eval("lambda p: p in {}".format(unitdef["skipped prefixes"]))
            else:
                skipfcn = None
            self.add_unit(
                symbols=unitdef["symbols"],
                sidict=self._get_si_dict(quant),
                scale_factor=unitdef["scale factor"],
                representative_symbol=unitdef["representative symbol"],
                create_metric_prefixes_for=unitdef["metric prefixes for"],
                unit_category=unitdef["category"],
                metric_skip_function=skipfcn,
            )

    def _get_si_dict(self, quant):
        """Get the si dictionary for quant."""
        si_symbols = ["m", "kg", "s", "A", "K", "cd", "mol"]
        res = {}
        unitlist = quant.units()
        for ii, si_sym in enumerate(si_symbols):
            if unitlist[ii] != 0:
                res[si_sym] = unitlist[ii]
        return res

    def quantity_from_string(self, string):
        """Create a Quantity instance from the supplied string.

        The string has to be in the format that misu uses for string representations, i.e.
        the following works:

        1.0 m
        1 m
        1 m^2 s^-1
        1 m/s
        1.248e+05 m/s
        -1.158e+05 m/s kg

        """
        # empty string?
        if not string.strip() == "":
            # Multiplication: replace all whitespace surounded by a-z,A-Z,0-9 with *
            string = re.sub(r"([a-z,A-Z,0-9])(\s+)([a-z,A-Z,0-9])", r"\1*\3", string)

            # Exponentiation: replace all ^ with **
            string = re.sub(r"\^", r"**", string)

            # inject self
            string = re.sub(r"\b([a-z,A-Z])", r"self.\1", string)
            # print(string)

            res = None
            try:
                res = eval(string)
            except NameError:
                print("String {} not understood.".format(string))
                res = None
            except SyntaxError:
                print("String {} not understood.".format(string))
                res = None
        else:
            res = self.dimensionless
        return res


def units_to_this_ns(unit_namespace):
    """Add the units defined in unit_namespace to the callers scope.

    Use call this to have direct access to the units, i.e. in your module do:

        import misu
        u = misu.UnitNamespace()
        misu.units_to_this_ns(u)

    Note that this uses a bit of black magic...

    """
    stack = inspect.stack()
    try:
        locals_ = stack[1][0].f_locals
    finally:
        del stack
    for symb in unit_namespace.known_units:
        locals_[symb] = getattr(unit_namespace, symb)


# def temperature_value_from_celsius(celsius):
#     return (celsius - 273.15) * K


# def temperature_change_from_celsius(celsius):
#     # SI root units
#     return celsius * K


# def temperature_value_from_fahrenheit(fahrenheit):
#     return (fahrenheit + 459.67) * R


# def temperature_change_from_fahrenheit(fahrenheit):
#     return fahrenheit * R


# # This is a decorator that will ensure arguments match declared units
# def dimensions(**_params_):
#     def check_types(_func_, _params_=_params_):
#         def modified(*args, **kw):
#             if sys.version_info.major == 2:
#                 arg_names = _func_.func_code.co_varnames
#             elif sys.version_info.major == 3:
#                 arg_names = _func_.__code__.co_varnames
#             else:
#                 raise Exception("Invalid Python version!")
#             kw.update(zip(arg_names, args))
#             for name, category in _params_.items():
#                 param = kw[name]
#                 assert isinstance(
#                     param, Quantity
#                 ), """Parameter "{}" must be an instance of class Quantity
# (and must be of unit type "{}").""".format(
#                     name, category
#                 )
#                 assert (
#                     param.unitCategory() == category
#                 ), 'Parameter "{}" must be unit type "{}".'.format(
#                     name, category
#                 )
#             return _func_(**kw)

#         modified.__name__ = _func_.__name__
#         modified.__doc__ = _func_.__doc__
#         # Py 3 only
#         # modified.__annotations__ = _func_.__annotations__
#         return modified

#     # For IDEs, make sure the arg lists propagate through to the user
#     return check_types

if __name__ == "__main__":
    u = UnitNamespace()
    units_to_this_ns(u)
    print(m)
    u.quantity_from_string("1 m^-1 s")
    print(5 * mHz)
    print(5 * N)
    a=5*V
    print(a.units())

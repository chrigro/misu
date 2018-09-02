# coding=utf8
from __future__ import division, print_function
from collections import OrderedDict  # OrderedDict: from py 2.7 on
import sys
import re
import functools
import json
import inspect
import os.path as osp
from misu.siprefixes import siprefixes_sym

import misu.engine as engine
from misu.engine import Quantity, QuantityNP, ESignatureAlreadyRegistered


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
            unit_category="Length",
            representative_symbol="m",
            create_metric_prefixes_for=["m"],
            metric_skip_function=None,
        )

        # kg (special since prefix already included)
        self.add_unit(
            symbols=["g", "grams", "gram"],
            sidict=dict(kg=1.0),
            scale_factor=1.0,
            unit_category="Mass",
            representative_symbol="",
            create_metric_prefixes_for=["g"],
            metric_skip_function=None,
        )
        self.g.setRepresent(self.kg, "kg")

        # seconds
        self.add_unit(
            symbols=["s", "second", "sec", "seconds", "secs"],
            sidict=dict(s=1.0),
            scale_factor=1.0,
            unit_category="Time",
            representative_symbol="s",
            create_metric_prefixes_for=["s"],
            metric_skip_function=lambda p: p in ["a"],  # no "as" since it is a keyword
        )

        # ampere
        self.add_unit(
            symbols=["A", "ampere", "amperes", "amp", "amps"],
            sidict=dict(A=1.0),
            scale_factor=1.0,
            unit_category="Electric current",
            representative_symbol="A",
            create_metric_prefixes_for=["A"],
            metric_skip_function=None,
        )

        # ampere
        self.add_unit(
            symbols=["K", "kelvin"],
            sidict=dict(K=1.0),
            scale_factor=1.0,
            unit_category="Temperature",
            representative_symbol="K",
            create_metric_prefixes_for=["K"],
            metric_skip_function=None,
        )

        # candela
        self.add_unit(
            symbols=["cd", "candela", "ca"],
            sidict=dict(cd=1.0),
            scale_factor=1.0,
            unit_category="Luminous intensity",
            representative_symbol="cd",
            create_metric_prefixes_for=["cd"],
            metric_skip_function=None,
        )

        # mol
        self.add_unit(
            symbols=["mol", "mole", "moles"],
            sidict=dict(mol=1.0),
            scale_factor=1.0,
            unit_category="Ammount of substance",
            representative_symbol="mol",
            create_metric_prefixes_for=["mol"],
            metric_skip_function=None,
        )

        # create derived units according to the context
        self._create_derived_units(context)

    def add_unit(
        self,
        symbols,
        sidict,
        scale_factor,
        unit_category="",
        representative_symbol="",
        create_metric_prefixes_for=[],
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

        unit_category : string (default: "")
            Category the unit belongs to. A unit category is defined by having a unique 
            decomposition in SI base units. This should be specified for the first unit 
            per category only. Otherwise a ESignatureAlreadyRegistered is triggered, 
            which is caugth and translated into a warning.

        representative_symbol : string (default: "")
            Symbol that should be used to represent the unit in a result of a calculation.
            Must be in symbols. Note that the each value here overides the previously set
            representative symbol for the unit category (see unit_category above).

        create_metric_prefixes_for : list of unit symbols (default: empty list)
            List of symbols (must also be in symbols) to create derived units with the
            metric prefixes.

        metric_skip_function : callable (default: None)
            Callable that returns true for the metric prefixes for which the prefixed
            unit should not be created.

        """
        # First define the quantity based on si
        quantity = engine.Quantity(scale_factor)
        quantity.setValDict(sidict)
        # Add to registry and namespace
        for symbol in symbols:
            symbol = symbol.strip()
            if symbol == "":
                continue
            self._add_quant_attr(symbol, quantity)
            self._add_to_registry(symbol, quantity)
        # Add category
        if not unit_category == "":
            # print("Adding type {} to category {}".format(quantity.units(), unit_category))
            try:
                engine.addType(quantity, str(unit_category))
            except ESignatureAlreadyRegistered as e:
                print(
                    "WARNING: Can not resister {} for unit category {}: {}".format(
                        symbols[0], unit_category, e
                    )
                )
        # Set representative symbol
        if not representative_symbol == "":
            self._check_represent(representative_symbol, symbols)
            quantity.setRepresent(as_unit=quantity, symbol=representative_symbol)
        # Metric prefixes
        if not create_metric_prefixes_for == []:
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
            unitdefs = json.load(f, object_pairs_hook=OrderedDict)
        for unitdef in unitdefs.values():
            conts = unitdef["used in contexts"]
            if ("all" in conts) or (context in conts):
                quant = self.quantity_from_string(
                    unitdef["representation in SI or earlier defined unit"]
                )
                # optional stuff
                if "category" in unitdef.keys():
                    cat = unitdef["category"]
                else:
                    cat = ""
                if "representative symbol" in unitdef.keys():
                    rep_sym = unitdef["representative symbol"]
                else:
                    rep_sym = ""
                if "metric prefixes for" in unitdef.keys():
                    prefixes = unitdef["metric prefixes for"]
                else:
                    prefixes = []
                if "skipped prefixes" in unitdef.keys():
                    if unitdef["skipped prefixes"]:
                        skipfcn = eval(
                            "lambda p: p in {}".format(unitdef["skipped prefixes"])
                        )
                    else:
                        skipfcn = None
                # add the unit
                self.add_unit(
                    symbols=unitdef["symbols"],
                    sidict=self._get_si_dict(quant),
                    scale_factor=unitdef["scale factor"],
                    unit_category=cat,
                    representative_symbol=rep_sym,
                    create_metric_prefixes_for=prefixes,
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


# ----- helpers -----


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


# ----- decorators -----


def noquantity(func):
    """Decorator to assure the input parameters are no Quantity.

    Notes
    -----
    Usage example:

        @noquantity
        def example(a, b):
            # do something

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # call the wrapped function
        for arg in args:
            if isinstance(arg, (Quantity, QuantityNP)):
                raise TypeError("Quantity arguments not allowed.")
        for arg in kwargs.values():
            if isinstance(arg, (Quantity, QuantityNP)):
                raise TypeError("Quantity arguments not allowed.")
        res = func(*args, **kwargs)
        return res

    return wrapper


def calc_unitless(out_unit_list, **in_unit_kwargs):
    """Decorator to convert the input parameters to magnitude and the output
    parameter back to quantity.

    Parameters
    ----------
    out_unit_list : list of Quantities
        Units of the output in correct order.

    in_unit_kwargs : Quantities as keyword arguments
        Use this to specify the Quantities to which the input arguments should be
        converted to.

    Notes
    -----
    Usage example:

        @dimensions([u.m/u.s, u.kg], a=u.m, b=u.s, c=u.kg)
        def example(a, b, c=u.kg):
            # do something
            return a/b, c

    """

    def calc_unitless_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # names of the arguments
            if sys.version_info.major == 2:
                arg_names = func.func_code.co_varnames
            elif sys.version_info.major == 3:
                arg_names = func.__code__.co_varnames
            else:
                raise Exception("Invalid Python version!")
            # transfer args to kwargs
            kwargs.update(zip(arg_names, args))

            # now do the conversion
            if not in_unit_kwargs.keys() == kwargs.keys():
                raise ValueError(
                    "Unit keyword arguments must match the function parameter names."
                )
            conv_kwargs = {}
            for kk in in_unit_kwargs.keys():
                conv_kwargs[kk] = kwargs[kk].convert(in_unit_kwargs[kk])

            # call the function and convert the result.
            res = func(**conv_kwargs)
            if isinstance(res, (list, tuple)):
                if not len(res) == len(out_unit_list):
                    raise ValueError(
                        "Number of output quantities must match the number of the function return values."
                    )
                return [r * out_unit_list[jj] for jj, r in enumerate(res)]
            else:
                return res * out_unit_list[0]

        return wrapper

    return calc_unitless_decorator


# This is a decorator that will ensure arguments match declared unit category
def dimensions(**_params_):
    """Decorator to assure the parameters given have the correct unit category.

    Notes
    -----
    Usage example:

        @dimensions(a='Length', b='Time')
        def example(a, b):
            # do something

    """

    def check_types(_func_, _params_=_params_):
        def modified(*args, **kw):
            if sys.version_info.major == 2:
                arg_names = _func_.func_code.co_varnames
            elif sys.version_info.major == 3:
                arg_names = _func_.__code__.co_varnames
            else:
                raise Exception("Invalid Python version!")
            kw.update(zip(arg_names, args))
            for name, category in _params_.items():
                param = kw[name]
                assert isinstance(
                    param, Quantity
                ), """Parameter "{}" must be an instance of class Quantity
(and must be of unit type "{}").""".format(
                    name, category
                )
                assert (
                    param.unitCategory() == category
                ), 'Parameter "{}" must be unit type "{}".'.format(
                    name, category
                )
            return _func_(**kw)

        modified.__name__ = _func_.__name__
        modified.__doc__ = _func_.__doc__
        # Py 3 only
        # modified.__annotations__ = _func_.__annotations__
        return modified

    # For IDEs, make sure the arg lists propagate through to the user
    return check_types


# ----- helpers for quantities with offset -----

# We do not support quantities with offset. Here are some helpers for temperature values.


@noquantity
def k_val_from_c(celsius):
    kelvin = celsius - 273.15
    return kelvin


@noquantity
def c_val_from_k(kelvin):
    celsius = kelvin + 273.15
    return celsius


@noquantity
def k_val_from_f(fahrenheit):
    kelvin = (fahrenheit + 459.67) * 5 / 9
    return kelvin


@noquantity
def f_val_from_k(kelvin):
    fahrenheit = kelvin * 9 / 5 - 459.67
    return fahrenheit


@noquantity
def c_val_from_f(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    return celsius


@noquantity
def f_val_from_c(celsius):
    fahrenheit = celsius * 9 / 5 + 32
    return fahrenheit


if __name__ == "__main__":
    u = UnitNamespace("temperature")
    units_to_this_ns(u)
    print(m)
    u.quantity_from_string("1 m^-1 s")
    print(5 * mHz)
    print(5 * kg)
    a = 5 * kg
    print(a + 3 * g >> mg)

    tt = 4 * R
    print(tt)
    tt.setRepresent(R, "R")
    print(4 * R)
    tt.setRepresent(K, "K")
    print(4 * R)

    k_val_from_c(5)

    @dimensions(a="Length")
    def test(a):
        return a

    test(5 * m)

    @calc_unitless([u.m / u.s, u.m], a=u.m, b=u.s)
    def test2(a, b):
        return a / b, a

    print(test2(5 * u.m, 2 * u.s))

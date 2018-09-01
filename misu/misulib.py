# coding=utf8
from __future__ import division, print_function
import sys
import traceback
import math
import re
from misu.engine import *
from misu.SIprefixes import SIprefixes_sym


class UnitNamespace(object):

    def add_unit(self,
    symbols,
    quantity,
    mustCreateMetricPrefixes=False,
    valdict=None,
    unitCategory=None,
    metricSkipFunction=None,
    notes=None):
    """Add a unit to the namespace.

    symbols: string of space-delimited units.  These will also be eval'ed
                into the module namespace, and will be entered as keys in
                UnitRegistry.

    quantity: would typically be a result of a calculation against
                base SI or some other unit defined earlier.

    notes: any important notes about the unit.
    """
    if valdict:
        quantity.setValDict(valdict)
    first_symbol = symbols.strip().split(" ")[0].strip()
    if unitCategory:
        addType(quantity, unitCategory)
        quantity.setRepresent(as_unit=quantity, symbol=first_symbol)

    for i, symbol in enumerate(symbols.split(" ")):
        try:
            symbol = symbol.strip()
            if symbol == "":
                continue
            UnitRegistry[symbol] = quantity
            exec("global {s}; {s} = quantity".format(s=symbol))
            print("{s} put in globals".format(s=symbol))
        except:
            print(traceback.format_exc())

    # Metric prefixes
    if mustCreateMetricPrefixes:
        createMetricPrefixes(first_symbol, metricSkipFunction)




def createMetricPrefixes(symbol, skipfunction=None):
    """ Populates the namespace with all the SI-prefixed versions of the
    given symbol.  This uses exec() internally."""
    for prefix in SIprefixes_sym:
        if skipfunction and skipfunction(prefix):
            continue
        template = "global {p}{s}; {p}{s} = 1e{e} * {s}"
        subs = template.format(p=prefix, s=symbol, e=SIprefixes_sym[prefix].exponent)
        exec(subs)


# def createUnit(
#     symbols,
#     quantity,
#     mustCreateMetricPrefixes=False,
#     valdict=None,
#     unitCategory=None,
#     metricSkipFunction=None,
#     notes=None,
# ):
#     """
#         symbols: string of space-delimited units.  These will also be eval'ed
#                  into the module namespace, and will be entered as keys in
#                  UnitRegistry.

#         quantity: would typically be a result of a calculation against
#                   base SI or some other unit defined earlier.

#         notes: any important notes about the unit.
#     """
#     if valdict:
#         quantity.setValDict(valdict)
#     first_symbol = symbols.strip().split(" ")[0].strip()
#     if unitCategory:
#         addType(quantity, unitCategory)
#         quantity.setRepresent(as_unit=quantity, symbol=first_symbol)

#     for i, symbol in enumerate(symbols.split(" ")):
#         try:
#             symbol = symbol.strip()
#             if symbol == "":
#                 continue
#             UnitRegistry[symbol] = quantity
#             exec("global {s}; {s} = quantity".format(s=symbol))
#             print("{s} put in globals".format(s=symbol))
#         except:
#             print(traceback.format_exc())

#     # Metric prefixes
#     if mustCreateMetricPrefixes:
#         createMetricPrefixes(first_symbol, metricSkipFunction)


def quantity_from_string(string):
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
        res = dimensionless
    return res


def temperature_value_from_celsius(celsius):
    return (celsius - 273.15) * K


def temperature_change_from_celsius(celsius):
    # SI root units
    return celsius * K


def temperature_value_from_fahrenheit(fahrenheit):
    return (fahrenheit + 459.67) * R


def temperature_change_from_fahrenheit(fahrenheit):
    return fahrenheit * R


# This is a decorator that will ensure arguments match declared units
def dimensions(**_params_):
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



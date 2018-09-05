# Uses py.test.
from __future__ import print_function
import os
import sys

new_syspath = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(new_syspath)

import math
import numpy as np
import pytest

import misu

import pickle

import numpy
try:
    numpy.set_printoptions(formatter=dict(all=lambda x: '{:.3g}'.format(x)))
except:
    pass

u = misu.UnitNamespace('all')
const = misu.PhysConst(u)

a = 2.5 * u.kg / u.s
b = 34.67 * u.kg / u.s

a.setRepresent(u.kg/u.hr, 'kg/hr')
b.setRepresent(u.kg/u.hr, 'kg/hr')

def lookup_type(quantity):
    return 'Quantity: {} Type: {}'.format(quantity, quantity.unitCategory())


def test_representation():
    print(repr(a))
    assert repr(a) == '9000 kg/hr'
    assert repr(b) == '1.248e+05 kg/hr'


def test_format_simple():
    assert '{:.2f}'.format(b) == '124812.00 kg/hr'


@pytest.mark.xfail(reason='This requires further work.')
def test_format_left_align():
    fmtstr = '{:<20.2f}'.format(b)
    assert fmtstr == '124812.00 kg/hr     '


@pytest.mark.xfail(reason='This requires further work.')
def test_format_right_align():
    fmtstr = '{:>20.2f}'.format(b)
    assert fmtstr == '     124812.00 kg/hr'


def test_from_string1():
    u.quantity_from_string('1 m^2 s^-1') == 1 * u.m**2/u.s


def test_from_string2():
    u.quantity_from_string('1 m^2     s^-1') == 1 * u.m**2/u.s


def test_from_string3():
    assert u.quantity_from_string('-1.158e+05 m/s kg^6.0') == -1.158e+05* u.m/u.s* u.kg**6.0


def test_addition():
    assert repr(a + b) == '1.338e+05 kg/hr'


def test_subtraction():
    assert repr(a - b) == '-1.158e+05 kg/hr'


def test_multiplication():
    assert '{:.3f}'.format(a * b) == '86.675 kg^2.0 s^-2.0'


def test_division():
    assert '{:.5f}'.format(a / b) == '0.07211'


def test_comparison_smaller():
    assert 1*u.m < 2*u.m

def test_comparison_smaller_equal():
    assert 3*u.m <= 3*u.m

def test_comparison_equal():
    assert 3*u.m == 3*u.m

def test_comparison_larger():
    assert 3*u.m > 2*u.m

def test_comparison_larger_equal():
    assert 3*u.m >= 3*u.m

def test_comparison_equal():
    assert 3*u.m == 3*u.m

def test_incompatible_units():
    with pytest.raises(misu.EIncompatibleUnits) as E:
        x = 2.0*u.m + 3.0*u.kg
    print(str(E.value))
    assert str(E.value) == 'Incompatible units: 2 m and 3 kg'


def test_operator_precedence():
    assert repr(2.0 * u.kg / u.s * 3.0) == '2.16e+04 kg/hr'
    assert repr(2.0 * 3.0 * u.kg / u.s) == '2.16e+04 kg/hr'
    assert repr((1.0/u.m)**0.5) == '1.0 m^-0.5'

    assert repr(((u.kg ** 2.0)/(u.m))**0.5) == '1.0 m^-0.5 kg^1.0'
    # assert repr((1.56724 * (u.kg ** 2.0)/(u.m * (u.s**2.0)))**0.5) \
    assert '{:.2f}'.format((1.56724 * (u.kg ** 2.0)/(u.m * (u.s**2.0)))**0.5) \
        == '1.25 m^-0.5 kg^1.0 s^-1.0'


def test_si_prefixes():
    assert 'Hz = {}'.format(u.Hz) == 'Hz = 1 Hz'
    assert 'kHz = {}'.format(u.kHz) == 'kHz = 1000 Hz'
    assert 'MHz = {}'.format(u.MHz) == 'MHz = 1e+06 Hz'
    assert 'GHz = {}'.format(u.GHz) == 'GHz = 1e+09 Hz'


def test_conversions():
    assert 1*u.m >> u.ft == 3.280839895013123
    assert 100*u.kg >> u.lb == 220.46226218487757


def test_unit_category():
    assert lookup_type(u.BTU) == 'Quantity: 1054 J Type: Energy'
    assert lookup_type(u.lb) == 'Quantity: 0.4536 kg Type: Mass'
    assert lookup_type(200 * u.MW * 10 * u.d) \
           == 'Quantity: 1.728e+14 J Type: Energy'


def test_func_decorator1():
    """ This tests the function decorator on Reynolds number,
    a standard dimensionless number in process engineering."""

    # Function definition
    @misu.dimensions(rho='Mass density', v='Velocity', L='Length',
                mu='Dynamic viscosity')
    def Reynolds_number(rho, v, L, mu):
        return rho * v * L / mu

    # Test 1
    data = dict(
        rho=1000*u.kg/u.m3,
        v=12*u.m/u.s,
        L=5*u.inch,
        mu=1e-3*u.Pa*u.s)
    assert 'Re = {}'.format(Reynolds_number(**data)) == 'Re = 1524000.0'

    # Test 2
    data = dict(
        rho=1000*u.kg/u.m3,
        v=12*u.m/u.s,
        L=1.5*u.inch,
        mu=1.011e-3*u.Pa*u.s)
    Re = Reynolds_number(**data)
    assert 'Re = {:.2e}'.format(Re) == 'Re = 4.52e+05'

    # Friction factor is another engineering quantity.
    # The Colebrook equation requires iteration, but there
    # are various approximations to the Colebrook equation
    # that do not require iteration, like Haaland below.
    @misu.dimensions(roughness='Length', Dh='Length', Re='Dimensionless')
    def friction_factor_Colebrook(roughness, Dh, Re):
        '''Returns friction factor.
        http://hal.archives-ouvertes.fr/docs/00/33/56/55/PDF/fast_colebrook.pdf
        '''
        K = roughness.convert(u.m) / Dh.convert(u.m)
        l = math.log(10)
        x1 = l * K * Re / 18.574
        x2 = math.log(l * Re.magnitude / 5.02)
        zj = x2 - 1./5.
        # two iterations
        for i in range(2):
            ej = (zj + math.log(x1 + zj) - x2) / (1. + x1 + zj)
            tol = (1. + x1 + zj + (1./2.)*ej) * ej * (x1 + zj) \
                / (1. + x1 + zj + ej + (1./3.)*ej**2)
            zj = zj - tol

        return (l / 2.0 / zj)**2

    @misu.dimensions(roughness='Length', Dh='Length', Re='Dimensionless')
    def friction_factor_Colebrook_Haaland(roughness, Dh, Re):
        K = roughness.convert(u.m) / Dh.convert(u.m)
        tmp = math.pow(K/3.7, 1.11) + 6.9 / Re
        inv = -1.8 * math.log10(tmp)
        return u.dimensionless * (1./inv)**2

    f = friction_factor_Colebrook(1e-6*u.m, 1.5*u.inch, Re)
    fH = friction_factor_Colebrook_Haaland(1e-6*u.m, 1.5*u.inch, Re)
    assert 'At Re = {:.2f}, friction factor = {:.5f}'.format(Re, f) \
        == 'At Re = 452225.52, friction factor = 0.01375'
    assert 'At Re = {:.2f}, friction factorH = {:.5f}'.format(Re, fH) \
        == 'At Re = 452225.52, friction factorH = 0.01359'

    assert 'f.unitCategory() = {}'.format(f.unitCategory()) \
        == 'f.unitCategory() = Dimensionless'
    assert 'fH.unitCategory() = {}'.format(fH.unitCategory()) \
        == 'fH.unitCategory() = Dimensionless'

    # The friction factor can then be used to calculate the
    # expected drop in pressure produced by flow through a
    # pipe.
    @misu.dimensions(
        fD='Dimensionless',
        D='Length',
        rho='Mass density',
        v='Velocity',
        L='Length')
    def pressure_drop(fD, D, rho, v, L=1*u.m):
        '''Arguments are
            fD:  Darcy-Weisbach friction factor
            L:   Length of pipe (default 1 metre)
            D:   Diameter of pipe
            rho: Density of the fluid
            v:   Velocity of the fluid
        '''
        return fD * L / D * rho * v**2 / 2

    # Test the pressure drop
    flow = 1*u.m3/u.s
    u.m.setRepresent(as_unit=u.inch, symbol='"')
    u.Pa.setRepresent(as_unit=u.bar, symbol='bar')
    lines = []
    for D in [x*u.inch for x in range(1, 11)]:
        v = flow / D**2 / math.pi * 4
        rho = 1000*u.kg/u.m3
        Re = Reynolds_number(rho=rho, v=v, L=D, mu=1e-3*u.Pa*u.s)
        f = friction_factor_Colebrook(1e-5*u.m, D, Re)
        lines.append('Pressure drop at diameter {} = {}'.format(
            D, pressure_drop(f, D, rho, v, L=1*u.m)))

    # Spot checks
    assert lines[0] == 'Pressure drop at diameter 1 " = 1.215e+04 bar'
    assert lines[4] == 'Pressure drop at diameter 5 " = 2.865 bar'
    assert lines[9] == 'Pressure drop at diameter 10 " = 0.08282 bar'


def test_func_decorator2():
    # Working only in m
    u.m.setRepresent(as_unit=u.m, symbol='m')

    @misu.dimensions(x='Length')
    def f(x, y, z):
        return x*y*z

    assert f(12*u.cm, 1, 1) == 0.12*u.m

    @misu.dimensions(y='Length')
    def f(x, y, z):
        return x*y*z

    assert f(1, 12*u.cm, 1) == 0.12*u.m

    @misu.dimensions(z='Length')
    def f(x, y, z):
        return x*y*z

    assert f(1, 1, 12*u.cm) == 0.12*u.m

def test_pickle():
    var = 2.5 * u.kg / u.s
    pick = pickle.dumps(var)
    res = pickle.loads(pick)
    assert var==res

def test_pickle_numpy():
    var = np.array([2.5, 4]) * u.kg / u.s
    pick = pickle.dumps(var)
    res = pickle.loads(pick)
    assert (var[0]==res[0] and var[1]==res[1])

def test_numpy_multiplication():
    x1 = u.kg * np.array([1, 2, 3])
    x2 = np.array([1, 2, 3]) * u.s
    x3 = 5 * u.kg
    x4 = u.m * 5
    x5 = x1 * x2
    x6 = x3 * x4
    assert repr(x1) == '[1 2 3] kg'
    assert repr(x2) == '[1 2 3] s'
    assert repr(x3) == '5 kg'
    assert repr(x4) == '5 m'
    assert repr(x5) == '[1 4 9] kg^1.0 s^1.0'
    assert repr(x6) == '25.0 m^1.0 kg^1.0'

def test_numpy_sum():
    x1 = 1*u.kg + np.array([1, 2, 3])*u.kg
    x2 = np.array([1, 2, 3])*u.kg + 1*u.kg
    x3 = np.array([1, 2, 3])*u.kg + np.array([1, 2, 3])*u.kg
    assert repr(x1) == '[2 3 4] kg'
    assert repr(x2) == '[2 3 4] kg'
    assert repr(x3) == '[2 4 6] kg'

def test_numpy_division():
    x1 = np.array([1, 2, 3])/(4*u.s)
    x2 = np.array([1, 2, 3])*u.kg / (2*u.kg)
    x3 = 1*u.kg / np.array([1, 2, 3])
    x4 = 1 / np.array([1, 2, 3]) / u.m
    assert repr(x1) == '[0.25 0.5 0.75] Hz'
    assert repr(x2) == '[0.5 1 1.5]'
    assert repr(x3) == '[1 0.5 0.333] kg'
    assert repr(x4) == '[1 0.5 0.333] m^-1.0'

def test_numpy_operations():
    x = np.array([1, 2, 3]) * u.kg
    y = x / (20*u.minutes)
    assert repr(y) == '[3 6 9] kg/hr'
    assert repr(y**2) \
        == '[6.94e-07 2.78e-06 6.25e-06] kg^2.0 s^-2.0'


def test_numpy_addition():
    x = np.array([1, 2, 3]) * u.kg
    y = np.array([1, 2, 3]) * u.lb
    assert repr(x+y) == '[1.45 2.91 4.36] kg'
    lbval = x+y >> u.lb
    assert np.allclose(lbval,
        np.array([3.20462262,  6.40924524,  9.61386787]))


def test_numpy_subtraction():
    x = np.array([1, 2, 3]) * u.kg
    y = np.array([1, 2, 3]) * u.lb
    assert repr(x-y) == '[0.546 1.09 1.64] kg'


def test_numpy_slice():
    x = np.array([ 0.08400557, 0.19897197, 0.12407021, 0.11867142]) * u.kg/u.hr
    assert repr(x[:2]) == '[0.084 0.199] kg/hr'
    assert repr(x[3]) == '0.1187 kg/hr'

def test_numpy_sin():
    mags = np.array([ 0.08400557, 0.19897197, 0.12407021, 0.11867142])
    x = mags * u.kN
    assert np.allclose(np.sin(mags) , np.sin(x/u.kN))
    assert np.allclose(np.sin(mags), np.sin(x >> u.kN))


if __name__ == '__main__':
    test_numpy_division()

# -*- coding: utf-8 -*-
# cython: profile=True
"""
Created on Wed Sep 04 22:15:13 2013

@author: Caleb Hattingh
"""

cimport cython
import numpy as np
cimport numpy as np
from cpython.array cimport array, copy


class EIncompatibleUnits(Exception):
    pass


class ESignatureAlreadyRegistered(Exception):
    pass


cdef class Quantity


cdef inline int isquantity(var):
    ''' Checks whether var is an instance of type 'Quantity'.
    Returns True or False.'''
    return isinstance(var, Quantity)


cdef inline void copyunits(Quantity source, Quantity dest, float power):
    ''' Cython limitations require that both source and dest are the same
    type. '''
    cdef int i
    for i from 0 <= i < 7:
        dest.unit[i] = source.unit[i] * power


QUANTITYTYPE = {}
cpdef addtype(Quantity q, str name):
    if q.unit_as_tuple() in QUANTITYTYPE:
        raise ESignatureAlreadyRegistered('The unit {} already registered, owned by: {}'.format(
            str(q.unitstring()), QUANTITYTYPE[q.unit_as_tuple()]))
    QUANTITYTYPE[q.unit_as_tuple()] = name


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Quantity assertquantity(x):
    if isquantity(x):
        return x
    else:
        return Quantity.__new__(Quantity, x)


ctypedef double[7] uarray


cdef list symbols = ['m', 'kg', 's', 'A', 'K', 'cd', 'mol']


cdef inline sameunits(Quantity self, Quantity other):
    cdef int i
    for i from 0 <= i < 7:
        if self.unit[i] != other.unit[i]:
            raise EIncompatibleUnits(
                'Incompatible units: {} and {}'.format(self, other))


REPRESENTCACHE = {}

# The unit registry is a lookup list where you can find a specific
# UnitDefinition from a particular symbol.  Note that multiple entries
# in the UNITREGISTRY can point to the same unit definition, because
# there can be many synonyms for a particular unit, e.g.
# s, sec, secs, seconds
UNITREGISTRY = {}


@cython.freelist(8)
cdef class Quantity:
    cdef readonly double _magnitude
    cdef readonly np.ndarray _magnitudeNP
    cdef readonly int mag_is_array
    cdef uarray unit
    __array_priority__ = 20.0

    def __cinit__(self, magnitude):
        if not isinstance(magnitude, np.ndarray):
            self.mag_is_array = 0
            self._magnitude = magnitude
        else:
            self.mag_is_array = 1
            self._magnitudeNP = magnitude
        self.unit[:] = [0,0,0,0,0,0,0]

    @property
    def magnitude(self):
        if self.mag_is_array == 0:
            return self._magnitude
        else:
            return self._magnitudeNP

    # ------ Pickling support ------

    def __reduce__(self):
        return (Quantity, (self.magnitude,), self.getunit(), None, None)

    def __setstate__(self, unit):
        self.setunit(unit)

    # ------ End pickling support ------

    cdef inline tuple unit_as_tuple(self):
        return tuple(self.units())

    def setunitdict(self, dict valdict):
        cdef int i
        cdef list values
        values = [valdict.get(s) or 0 for s in symbols]
        for i from 0 <= i < 7:
            self.unit[i] = values[i]

    def getunit(self):
        cdef list out
        cdef int i
        out = [0.0]*7
        for i from 0 <= i < 7:
            out[i] = self.unit[i]
        return out

    def setunit(self, list unit):
        cdef int i
        for i from 0 <= i < 7:
            self.unit[i] = unit[i]

    def setrepresent(self, as_unit=None, symbol='',
        convert_function=None, format_spec='.4g'):
        '''By default, the target representation is arrived by dividing
        the current unit MAGNITUDE by the target unit MAGNITUDE, and
        appending the desired representation symbol.

        However, if a conversion_function is supplied, then INSTEAD the
        conversion function will be called so:

            output_magnitude = conversion_function(self.magnitude)
            output_symbol = symbol

            result = '{} {}'.format(output_magnitude, output_symbol)

        The intention of the function argument is to allow
        non-proportional conversion, typically temperature but also things
        like barg, bara, etc.

        Note that if a convert_function is supplied, the as_unit arg
        is IGNORED.'''
        if not (as_unit or convert_function):
            raise Exception('Either a target unit or a conversion function must be supplied.')

        if convert_function == None:
            def proportional_conversion(instance, _):
                return instance.convert(as_unit)
            convert_function = proportional_conversion
        REPRESENTCACHE[self.unit_as_tuple()] = dict(
            convert_function=convert_function,
            symbol=symbol,
            format_spec=format_spec)

    def units(self):
        cdef list out = []
        cdef int i
        for i in range(7):
            out.append(self.unit[i])
        return out

    cpdef tuple as_tuple(self):
        if self.mag_is_array == 0:
            return (self._magnitude, self.unit_as_tuple())
        else:
            raise Exception("Only available for scalar Quantities.")

    def unitstring(self):
        if self.unit_as_tuple() in REPRESENTCACHE:
            r = REPRESENTCACHE[self.unit_as_tuple()]
            ret = '{}'.format(r['symbol'])
            return ret
        else:
            text = ' '.join(['{}^{}'.format(k,v) for k, v in zip(symbols, self.units()) if v != 0])
            ret = '{}'.format(text)
            return ret

    def _getmagnitude(self):
        if self.unit_as_tuple() in REPRESENTCACHE:
            r = REPRESENTCACHE[self.unit_as_tuple()]
            return r['convert_function'](self, self.magnitude)

        else:
            return self.magnitude

    def _getsymbol(self):
        if self.unit_as_tuple() in REPRESENTCACHE:
            r = REPRESENTCACHE[self.unit_as_tuple()]
            return r['symbol']
        else:
            return self.unitstring()

    def _getrepresenttuple(self):
        if self.unit_as_tuple() in REPRESENTCACHE:
            r = REPRESENTCACHE[self.unit_as_tuple()]
            format_spec = r['format_spec']
        else:
            format_spec = ''
        # Temporary fix for a numpy display issue
        if self.mag_is_array == 1:
            format_spec = ''
        return self._getmagnitude(), self._getsymbol(), format_spec

    def unitcategory(self):
        if self.unit_as_tuple() in QUANTITYTYPE:
            return QUANTITYTYPE[self.unit_as_tuple()]
        else:
            msg = 'The collection of units: "{}" has not been defined as a category yet.'
            raise Exception(msg.format(str(self)))

    def __str__(self):
        mag, symbol, format_spec = self._getrepresenttuple()
        number_part = format(mag, format_spec)
        if symbol == '':
            return number_part
        else:
            return ' '.join([number_part, symbol])

    def __repr__(self):
        return str(self)

    def __format__(self, format_spec):
        # Ignore the stored format_spec, use the given one.
        mag, symbol, stored_format_spec = self._getrepresenttuple()
        if format_spec == '':
            format_spec = stored_format_spec
        number_part = format(mag, format_spec)
        if symbol == '':
            return number_part
        else:
            return ' '.join([number_part, symbol])

    def __float__(self):
        assert self.unitcategory() == 'Dimensionless', 'Must be dimensionless for __float__()'
        return self.magnitude

    # Arithmetric for standard python types.
    # See https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html
    # We also call them for numpy operations, see __array_ufunc__ below.

    @staticmethod
    def _add(x, y):
        cdef Quantity xq
        cdef Quantity yq
        cdef Quantity ans

        xq = assertquantity(x)
        yq = assertquantity(y)
        ans = Quantity.__new__(Quantity, xq.magnitude + yq.magnitude)
        sameunits(xq, yq)
        copyunits(xq, ans, 1)
        return ans

    def __add__(x, y):
        return Quantity._add(x, y)

    @staticmethod
    def _sub(x, y):
        cdef Quantity xq
        cdef Quantity yq
        cdef Quantity ans

        xq = assertquantity(x)
        yq = assertquantity(y)
        ans = Quantity.__new__(Quantity, xq.magnitude - yq.magnitude)
        sameunits(xq, yq)
        copyunits(xq, ans, 1)
        return ans

    def __sub__(x, y):
        return Quantity._sub(x, y)

    @staticmethod
    def _mul(x, y):
        cdef Quantity xq
        cdef Quantity yq
        cdef Quantity ans
        cdef int i

        xq = assertquantity(x)
        yq = assertquantity(y)
        ans = Quantity.__new__(Quantity, xq.magnitude * yq.magnitude)
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i] + yq.unit[i]
        return ans

    def __mul__(x, y):
        return Quantity._mul(x, y)

    @staticmethod
    def _div(x,y):
        cdef Quantity xq = assertquantity(x)
        cdef Quantity yq = assertquantity(y)
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude / yq.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i] - yq.unit[i]
        return ans

    def __div__(x,y):
        return Quantity._div(x, y)

    @staticmethod
    def _truediv(x, y):
        cdef Quantity xq = assertquantity(x)
        cdef Quantity yq = assertquantity(y)
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude / yq.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i] - yq.unit[i]
        return ans

    def __truediv__(x, y):
        return Quantity._truediv(x, y)

    @staticmethod
    def _pow(x, y, z):
        cdef Quantity xq = assertquantity(x)
        assert not isquantity(y), 'The exponent must not be a quantity!'
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude ** y)
        copyunits(xq, ans, y)
        return ans

    def __pow__(x, y, z):
        return Quantity._pow(x, y, z)

    @staticmethod
    def _neg(x):
        cdef Quantity ans = Quantity.__new__(Quantity, -x.magnitude)
        copyunits(x, ans, 1)
        return ans

    def __neg__(x):
        return Quantity._neg(x)

    @staticmethod
    def _pos(x):
        cdef Quantity ans = Quantity.__new__(Quantity, x.magnitude)
        copyunits(x, ans, 1)
        return ans

    def __pos__(x):
        return Quantity._pos(x)

    @staticmethod
    def _richcmp(x, y, int op):
        """
        <   0
        <=  1
        ==  2
        !=  3
        >   4
        >=  5
        """
        cdef Quantity xq = assertquantity(x)
        cdef Quantity yq = assertquantity(y)
        sameunits(xq, yq)
        if op == 0:
            return xq.magnitude < yq.magnitude
        elif op == 1:
            return xq.magnitude <= yq.magnitude
        elif op == 2:
            return xq.magnitude == yq.magnitude
        elif op == 3:
            return xq.magnitude != yq.magnitude
        elif op == 4:
            return xq.magnitude > yq.magnitude
        elif op == 5:
            return xq.magnitude >= yq.magnitude

    def __richcmp__(x, y, int op):
        return Quantity._richcmp(x, y, op)

    def __abs__(x):
        """Numpy abs is different"""
        cdef Quantity ans = Quantity.__new__(Quantity, abs(x.magnitude))
        copyunits(x, ans, 1)
        return ans

    @staticmethod
    def _rshift(x, y):
        return x.convert(y)

    def __rshift__(x, y):
        """ Use quantity1 >> quantity2 to get the value of quantity1 in quantity2"""
        return Quantity._rshift(x, y)

    def convert(self, Quantity target_unit):
        assert isquantity(target_unit), 'Target must be a quantity.'
        assert target_unit.mag_is_array == 0, 'Target must be scalar not an array.'
        sameunits(self, target_unit)
        return self.magnitude / target_unit.magnitude

    @staticmethod
    def _check_dimensionless(x):
        if x.unitcategory() != 'Dimensionless':
            raise EIncompatibleUnits('Argument must be dimensionless.')

    # ------ Implement numpy functionality ------

    def copy(self):
        cdef Quantity ans = Quantity.__new__(Quantity, self.magnitude)
        copyunits(self, ans, 1)
        return ans

    def __getitem__(self, val):
        """Slicing for ndarray valued Quantities"""
        if self.mag_is_array == 0:
            raise TypeError("Scalar quantity not subscriptable")
        cdef Quantity ans
        ans = Quantity.__new__(Quantity, self.magnitude[val])
        copyunits(self, ans, 1)
        return ans

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support numpy ufuncs.

        See https://docs.scipy.org/doc/numpy-1.15.0/reference/ufuncs.html#ufuncs
        and https://github.com/numpy/numpy/blob/v1.15.1/numpy/lib/mixins.py#L63-L183

        Notes:
        ------
        Not all ufuncs are implemented. Implementing the rest is straigth forward, but 
        will be quite verbose.

        """

        cdef Quantity ans

        ufuncs_1inp_dimensionless_in_out = ( 'sin', 'cos', 'tan', 'arcsin', 'arccos',
                'arctan', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                'deg2rad', 'rad2deg', 'exp', 'exp2', 'log', 'log2', 'log10', 'expm1',
                'log1p')

        ufuncs_1inp_nounit_out = ('sign', 'isfinite', 'isinf', 'isnan', 'isnat', 'fabs',
                'signbit')

        ufuncs_1inp_same_unit_in_out = ('absolute', 'fabs', 'rint', 'floor', 'ceil',
                'trunc')

        ufuncs_2inp_same_unit_in_out = ('maximum', 'minimum', 'fmin', 'fmax')

        ufuncs_2inp_no_unit_out = ('greater', 'greater_equal', 'less', 'less_equal',
                'not_equal', 'equal', 'logical_and', 'logical_or', 'logical_xor',
                'logical_not')

        print ufunc.__name__
        if ufunc.__name__ == 'add':
            return Quantity._add(*inputs)
        if ufunc.__name__ == 'subtract':
            return Quantity._sub(*inputs)
        if ufunc.__name__ == 'multiply':
            return Quantity._mul(*inputs)
        elif ufunc.__name__ == 'true_divide':
            return Quantity._truediv(*inputs)
        elif ufunc.__name__ == 'divide':
            return Quantity._div(*inputs)
        elif ufunc.__name__ == 'negative':
            return Quantity._neg(*inputs)
        elif ufunc.__name__ == 'positive':
            return Quantity._pos(*inputs)
        elif ufunc.__name__ == 'power':
            return Quantity._pow(*inputs)
        elif ufunc.__name__ == 'right_shift':
            return Quantity._rshift(*inputs)
        else:
            if 'out' in kwargs:
                raise NotImplementedError("Keyword argument 'out' not supported.")
            if ufunc.__name__ in ufuncs_1inp_dimensionless_in_out:
                Quantity._check_dimensionless(inputs[0])
                ans = Quantity.__new__(Quantity, getattr(ufunc, method)(inputs[0].magnitude, **kwargs))
                return ans
            elif ufunc.__name__ in ufuncs_1inp_nounit_out:
                return getattr(ufunc, method)(inputs[0].magnitude, **kwargs)
            elif ufunc.__name__ in ufuncs_1inp_same_unit_in_out:
                ans = Quantity.__new__(Quantity, getattr(ufunc, method)(inputs[0].magnitude, **kwargs))
                copyunits(self, ans, 1)
                return ans
            elif ufunc.__name__ in ufuncs_2inp_same_unit_in_out:
                sameunits(inputs[0], inputs[1])
                ans = Quantity.__new__(Quantity, getattr(ufunc, method)(inputs[0].magnitude, inputs[1].magnitude, **kwargs))
                copyunits(self, ans, 1)
                return ans
            elif ufunc.__name__ in ufuncs_2inp_no_unit_out:
                sameunits(inputs[0], inputs[1])
                return getattr(ufunc, method)(inputs[0].magnitude, inputs[1].magnitude, **kwargs)
            elif ufunc.__name__ =='sqrt':
                ans = Quantity.__new__(Quantity, getattr(ufunc, method)(inputs[0].magnitude, **kwargs))
                copyunits(self, ans, 0.5)
                return ans
            elif ufunc.__name__ =='square':
                ans = Quantity.__new__(Quantity, getattr(ufunc, method)(inputs[0].magnitude, **kwargs))
                copyunits(self, ans, 2)
                return ans
            elif ufunc.__name__ =='cbrt':
                ans = Quantity.__new__(Quantity, getattr(ufunc, method)(inputs[0].magnitude, **kwargs))
                copyunits(self, ans, 1./3.)
                return ans
            elif ufunc.__name__ =='reciprocal':
                ans = Quantity.__new__(Quantity, getattr(ufunc, method)(inputs[0].magnitude, **kwargs))
                copyunits(self, ans, -1)
                return ans
            else:
                return NotImplemented

    def __array__(self):
        if self.mag_is_array == 1:
            return self._magnitudeNP
        else:
            return self._magnitude
        


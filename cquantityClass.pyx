# cython: profile=True
"""
Created on Wed Sep 04 22:15:13 2013

@author: Caleb
"""
#from __future__ import absolute_import, division, print_function, unicode_literals
cimport cython
#cimport cpython.array
from cpython.array cimport array, copy
#from cython.view import array

# Quantity Type
#
# This is a dict that defines a particular configuration of
# fundamental SI units (m, kg, etc) as a named quantity type.
# For example,
#
#   m^1: 'Length'
#   m^2: 'Area'
#   m^3: 'Volume'
#   m/s: 'Velocity'
#   kg/s: 'Mass flowrate'

# Forward declaration
cdef class Quantity

ctypedef fused Quantity_or_float:
    cython.float
    Quantity

QuantityType = {}
cpdef addType(Quantity q, char* name):
    if q.unit_as_tuple() in QuantityType:
        raise Exception('This unit def already registered, owned by: {}'.format(
            QuantityType[q.unit_as_tuple()]))
    QuantityType[q.unit_as_tuple()] = name

cdef inline Quantity assertQuantity(x):
    if isinstance(x, Quantity):
        return x
    else:
        return Quantity.__new__(Quantity, x)
#    if x is Quantity:
#        return x
#    else:
#        return Quantity.__new__(Quantity, <double>x)


cdef list symbols = ['m', 'kg', 's', 'A', 'K', 'ca', 'mole']

cdef inline int isQuantity(var):
    ''' checks whether var is an instance of type 'Quantity'.
    Returns True or False.'''
    return isinstance(var, Quantity)

RepresentCache = {}

# The unit registry is a lookup list where you can find a specific
# UnitDefinition from a particular symbol.  Note that multiple entries
# in the UnitRegistry can point to the same unit definition, because
# there can be many synonyms for a particular unit, e.g.
# s, sec, secs, seconds
UnitRegistry = {}
class UnitDefinition(object):
    def __init__(self, symbols, quantity, notes):
        self.symbols = [s.strip() for s in symbols.strip().split(' ') if s.strip() != '']
        self.quantity = quantity
        self.notes = notes
        for s in self.symbols:
            try:
                UnitRegistry[s] = self
                exec('global {s}; {s} = quantity'.format(s=s))
            except:
                print 'Error create UnitRegistry entry for symbol: {}'.format(s)
#                print
#                print ' '*4 + traceback.format_exc(0)
#                print

class EIncompatibleUnits(Exception):
    pass

cdef array _nou  = array('d', [0,0,0,0,0,0,0])

@cython.freelist(8)
cdef class Quantity:
    cdef readonly double magnitude
    cdef double unit[7]
    __array_priority__ = 20.0

    def __cinit__(self, double magnitude):
        self.magnitude = magnitude
        self.unit[:] = [0,0,0,0,0,0,0]

    cdef inline tuple unit_as_tuple(self):
        return tuple(self.units())
        #return tuple(self.unit.tolist())

    def setValDict(self, dict valdict):
        cdef int i
        cdef list values
        values = [valdict.get(s) or 0 for s in symbols]
        for i from 0 <= i < 7:
            self.unit[i] = values[i]

    def selfPrint(self):
        dict_contents = ','.join(['{}={}'.format(s,v) for s,v in dict(zip(symbols, self.units())).iteritems() if v != 0.0])
        return 'Quantity({}, dict({}))'.format(self.magnitude, dict_contents)

    def setRepresent(self, as_unit=None, symbol='',
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
        RepresentCache[self.unit_as_tuple()] = dict(
            convert_function=convert_function,
            symbol=symbol,
            format_spec=format_spec)

    def units(self):
        cdef list out = []
        cdef int i
        for i in range(7):
            out.append(self.unit[i])
        return out

    def _unitString(self):
        if self.unit_as_tuple() in RepresentCache:
            r = RepresentCache[self.unit_as_tuple()]
            ret = '{}'.format(r['symbol'])
            return ret
        else:
            text = ' '.join(['{}^{}'.format(k,v) for k, v in zip(symbols, self.units()) if v != 0])
            ret = '{}'.format(text)
            return ret

    def _getmagnitude(self):
        if self.unit_as_tuple() in RepresentCache:
            r = RepresentCache[self.unit_as_tuple()]
            return r['convert_function'](self, self.magnitude)
        else:
            return self.magnitude

    def _getsymbol(self):
        if self.unit_as_tuple() in RepresentCache:
            r = RepresentCache[self.unit_as_tuple()]
            return r['symbol']
        else:
            return self._unitString()

    def _getRepresentTuple(self):
        if self.unit_as_tuple() in RepresentCache:
            r = RepresentCache[self.unit_as_tuple()]
            mag = r['convert_function'](self, self.magnitude)
            symbol = r['symbol']
            format_spec = r['format_spec']
        else:
            mag = self.magnitude
            symbol = self._unitString()
            format_spec = ''
        # Temporary fix for a numpy display issue
        if not type(mag) in [float, int]:
            format_spec = ''
        return mag, symbol, format_spec

    def __str__(self):
        mag, symbol, format_spec = self._getRepresentTuple()
        number_part = format(mag, format_spec)
        if symbol == '':
            return number_part
        else:
            return ' '.join([number_part, symbol])

    def __repr__(self):
        return str(self)

    cdef inline sameunits(Quantity self, Quantity other):
        cdef int i
        for i from 0 <= i < 7:
            if self.unit[i] != other.unit[i]:            
                raise EIncompatibleUnits('Incompatible units: {} and {}'.format(self, other))

    def __add__(x, y):
        cdef Quantity xq = assertQuantity(x)
        cdef Quantity yq = assertQuantity(y)
        xq.sameunits(yq)
        #cdef Quantity ans = Quantity(xq.magnitude + yq.magnitude)
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude + yq.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i]
        return ans

    def __sub__(x, y):
        cdef Quantity xq = assertQuantity(x)
        cdef Quantity yq = assertQuantity(y)
        xq.sameunits(yq)
        #cdef Quantity ans = Quantity(xq.magnitude - yq.magnitude)
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude - yq.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i]
        return ans

    def unpack_or_default(self, other):
        try:
            return other.unit
        except:
            return _nou

    def __mul__(x, y):
        cdef Quantity xq = assertQuantity(x)
        cdef Quantity yq = assertQuantity(y)
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude * yq.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i] + yq.unit[i]
        #ans.unit = array('d', [x+y for x,y in zip(xq.unit, yq.unit)])
        return ans

    def __div__(x,y):
        cdef Quantity xq = assertQuantity(x)
        cdef Quantity yq = assertQuantity(y)
        #cdef Quantity ans = Quantity(xq.magnitude / yq.magnitude)
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude / yq.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i] - yq.unit[i]
        return ans

    def __truediv__(x, y):
        cdef Quantity xq = assertQuantity(x)
        cdef Quantity yq = assertQuantity(y)
#        if type(other.magnitude) == int:
#            denom = float(other.magnitude)
#        else:xq.unit
#            denom = other.magnitude
        #cdef Quantity ans = Quantity(xq.magnitude / yq.magnitude)
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude / yq.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i] - yq.unit[i]
        #ans.unit = array('d', [x-y for x,y in zip(xq.unit, yq.unit)])
        return ans

    def __pow__(x, y, z):
        cdef Quantity xq = assertQuantity(x)
        assert not isQuantity(y), 'The exponent must not be a quantity!'
        cdef Quantity ans = Quantity.__new__(Quantity, xq.magnitude ** y)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = xq.unit[i] * y
        #ans.unit = array('d', [x*y for x,y in zip(xq.unit, uvals)])
        return ans

    def __neg__(self):
        #cdef Quantity ans = Quantity(-self.magnitude)
        cdef Quantity ans = Quantity.__new__(Quantity, -self.magnitude)
        cdef int i
        for i from 0 <= i < 7:
            ans.unit[i] = self.unit[i]
        return ans

    def __cmp__(x, y):
        cdef Quantity xq = assertQuantity(x)
        cdef Quantity yq = assertQuantity(y)
        xq.sameunits(yq)
        if xq.magnitude < yq.magnitude:
            return -1
        elif xq.magnitude == yq.magnitude:
            return 0
        elif xq.magnitude > yq.magnitude:
            return 1
        else:
            raise Exception('Impossible.')

    def convert(self, target_unit):
        if isQuantity(target_unit):
            self.sameunits(target_unit)
            return self.magnitude / target_unit.magnitude
        elif isinstance(target_unit, str):
            target_unit_Q = eval(target_unit)
            self.sameunits(target_unit_Q)
            return '{} {}'.format(self.magnitude / target_unit_Q.magnitude, target_unit)

    def unitCategory(self):
        if self.unit_as_tuple() in QuantityType:
            return QuantityType[self.unit_as_tuple()]
        else:
            msg = 'The collection of units: "{}" has not been defined as a category yet.'
            raise Exception(msg.format(str(self)))

    def __format__(self, format_spec):
        # Ignore the stored format_spec, use the given one.
        mag, symbol, stored_format_spec = self._getRepresentTuple()
        if format_spec == '':
            format_spec = stored_format_spec
        number_part = format(mag, format_spec)
        if symbol == '':
            return number_part
        else:
#            if '.' in format_spec:
#                front, back = format_spec.split('.')
#                back = '.' + back
#                fstr = format(mag, back)
#                return ' '.join([fstr, symbol])
#            else:
#                fstr = str()
#
#            fstr = '{:' + format_spec + '} {}'
#            print fstr
#            return fstr.format(mag, symbol)
            return ' '.join([number_part, symbol])

    def __float__(self):
        assert self.unitCategory() == 'Dimensionless', 'Must be dimensionless for __float__()'
        return self.magnitude

    def __rshift__(self, other):
        return self.convert(other)

if __name__ == '__main__':
#    m = Quantity(1.0, {'m':1.0}, 'Length')
#    kg = Quantity(1.0, {'kg':1.0}, 'Mass')
#    rho = 1000*kg/m**3
#    print rho
    for i in xrange(100000):
        m = Quantity(1.0, {'m':1.0}, 'Length')
        kg = Quantity(1.0, {'kg':1.0}, 'Mass')
        rho = 1000*kg/m**3

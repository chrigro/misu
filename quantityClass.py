# -*- coding: utf-8 -*-
"""
Created on Wed Sep 04 22:15:13 2013

@author: Caleb
"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import collections
from struct import pack, unpack
import copy

dbg = 0

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
#
# Note that the key is the PACKED UNIT STRUCT (Ustruct)
QuantityType = {}
def addType(q, name):
    if q.unit in QuantityType:
        raise Exception('This unit def already registered, owned by: {}'.format(
            QuantityType[q.unit]))
    QuantityType[q.unit] = name


class QuantityTypeDefinition(object):
    def __init__(self, quantity):
        pass


# The Ustruct is a named tuple that stores an exponent for each
# fundamental SI unit.  Note that the exponents are not required to
# be integer.
symbols = ['m', 'kg', 's', 'A', 'K', 'ca', 'mole']
Ustruct = collections.namedtuple('Ustruct', ' '.join(symbols))

def isQuantity(var):
    ''' checks whether var is an instance of type 'Quantity'.
    Returns True or False.'''
    return isinstance(var, Quantity)

# This is a dictionary that keeps track of how specific Ustructs
# should be presented to the user.
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
                print
                print ' '*4 + traceback.format_exc(0)
                print

class EIncompatibleUnits(Exception):
    pass

class Quantity(object):
    fmt = 'f' * len(symbols)
    emptyunpacked = [0 for i in symbols]
    emptystruct = pack(fmt, *emptyunpacked)
    objectcache = collections.deque([]) # Use a list as a stack
    # Have to do this ourselves because of the detections
    # made in __del__()
    objectcache_maxlen = 10000

    __array_priority__ = 20.0

    def __new__(cls, magnitude, valdict=None, quantityTypeName=None):
        if len(cls.objectcache)>0:
            #import pdb; pdb.set_trace()
            pass
        try:
            x = cls.objectcache.pop()
        except IndexError:
            x = super(Quantity, cls).__new__(cls)            
        x.__init__(magnitude, valdict, quantityTypeName)
        x.cacheable = True # instance is cacheable
        #_cache[i] = x
        return x
        
    def __del__(self):
        # Instead of destruction, save the instance for reuse
        #import pdb; pdb.set_trace()
        if self.cacheable:
            if len(self.objectcache) == self.objectcache_maxlen:
                # Cache is full. Delete old entries.
                # TODO: must compare speed of pop vs popleft here.
                x = self.objectcache.popleft()
                x.cacheable = False # Must not enter objectcache on next __del__()
            self.objectcache.append(self)

    def __init__(self, magnitude, valdict=None, quantityTypeName=None):
        # This is slower:
        # lst=copy.copy(self.emptyunpacked)  # new list!
        lst=[0]*len(symbols)  # new list!
        if valdict:
            for k in valdict:
                lst[symbols.index(k)] = valdict[k]
        self.magnitude = magnitude
        self.unit = pack(self.fmt, *lst)
        # Add given quantityTypeName to external dict
        if quantityTypeName:
            #assert not self.unit in QuantityType, 'This category has already been declared.'
            QuantityType[self.unit] = quantityTypeName

    def selfPrint(self):
        #import pdb; pdb.set_trace()
        dict_contents = ','.join(['{}={}'.format(s,v) for s,v in self.units()._asdict().iteritems() if v != 0.0])
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
        RepresentCache[self.unit] = dict(
            convert_function=convert_function,
            symbol=symbol,
            format_spec=format_spec)

    def units(self):
        return Ustruct._make(unpack(self.fmt,self.unit))

    def _unitString(self):
        if self.unit in RepresentCache:
            r = RepresentCache[self.unit]
            return '{}'.format(r['symbol'])
        else:
            ut = self.units()._asdict()
            text = ' '.join(['{}^{}'.format(k,v) for k, v in ut.items() if v != 0])
            return '{}'.format(text)

    def _getmagnitude(self):
        if self.unit in RepresentCache:
            r = RepresentCache[self.unit]
            return r['convert_function'](self, self.magnitude)
        else:
            return self.magnitude

    def _getsymbol(self):
        if self.unit in RepresentCache:
            r = RepresentCache[self.unit]
            return r['symbol']
        else:
            return self._unitString()

    def _getRepresentTuple(self):
        if self.unit in RepresentCache:
            r = RepresentCache[self.unit]
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

    def sameunits(self, other):
        if not self.unit == other.unit:
            print self
            print other
            raise EIncompatibleUnits('Incompatible units: {} and {}'.format(self, other))

    def assertQuantity(self, other):
        if isQuantity(other):
            return other
        else:
            return Quantity(other)

    def __add__(self, other):
        other = self.assertQuantity(other)
        self.sameunits(other)
        ans = Quantity(self.magnitude + other.magnitude)
        ans.unit = copy.copy(self.unit)
        return ans

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self.assertQuantity(other)
        self.sameunits(other)
        ans = Quantity(self.magnitude - other.magnitude)
        ans.unit = copy.copy(self.unit)
        return ans

    def __rsub__(self, other):
        return  -1.0 * self.__sub__(other)

    def unpack_or_default(self, other):
        try:
            return unpack(self.fmt, other.unit)
        except:
            #return copy.copy(self.emptyunpacked)
            return [0]*len(symbols)        
#        if isQuantity(other):
#            return unpack(self.fmt, other.unit)
#        else:
#            #return copy.copy(self.emptyunpacked)
#            return [0]*len(symbols)

    def __mul__(self, other):
        if dbg==1:
            import pdb; pdb.set_trace()
        other = self.assertQuantity(other)
        ans = Quantity(self.magnitude * other.magnitude)
        uvals = self.unpack_or_default(other)
        ans.unit = pack(self.fmt, *[x+y for x,y in zip(unpack(self.fmt, self.unit), uvals)])
        return ans

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        other = self.assertQuantity(other)
        ans = Quantity(self.magnitude / other.magnitude)
        uvals = self.unpack_or_default(other)
        ans.unit = pack(self.fmt, *[x-y for x,y in zip(unpack(self.fmt, self.unit), uvals)])
        return ans

    def __rdiv__(self, other):
        other = self.assertQuantity(other)
        ans = Quantity(other.magnitude / self.magnitude)
        uvals = self.unpack_or_default(other)
        ans.unit = pack(self.fmt, *[y-x for x,y in zip(unpack(self.fmt, self.unit), uvals)])
        return ans

    def __truediv__(self, other):
        other = self.assertQuantity(other)
        if type(other.magnitude) == int:
            denom = float(other.magnitude)
        else:
            denom = other.magnitude
        ans = Quantity(self.magnitude / denom)
        uvals = self.unpack_or_default(other)
        ans.unit = pack(self.fmt, *[x-y for x,y in zip(unpack(self.fmt, self.unit), uvals)])
        return ans

    def __rtruediv__(self, other):
        other = self.assertQuantity(other)
        if type(other.magnitude) == int:
            numer = float(other.magnitude)
        else:
            numer = other.magnitude
        ans = Quantity(numer / self.magnitude)
        uvals = self.unpack_or_default(other)
        ans.unit = pack(self.fmt, *[y-x for x,y in zip(unpack(self.fmt, self.unit), uvals)])
        return ans

    def __pow__(self, other):
        #import pdb; pdb.set_trace()
        assert not isQuantity(other), 'The exponent must not be a quantity!'
        ans = Quantity(self.magnitude ** other)
        uvals = [other for x in symbols]
        ans.unit = pack(self.fmt, *[x*y for x,y in zip(unpack(self.fmt, self.unit), uvals)])
        return ans

    def __neg__(self):
        ans = Quantity(-self.magnitude)
        ans.unit = copy.copy(self.unit)
        return ans

    def __cmp__(self, other):
        other = self.assertQuantity(other)
        self.sameunits(other)
        if self.magnitude < other.magnitude:
            return -1
        elif self.magnitude == other.magnitude:
            return 0
        elif self.magnitude > other.magnitude:
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
        if self.unit in QuantityType:
            return QuantityType[self.unit]
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


.. image:: https://travis-ci.org/cjrh/misu.svg?branch=master
    :target: https://travis-ci.org/cjrh/misu

.. image:: https://coveralls.io/repos/github/cjrh/misu/badge.svg?branch=master
    :target: https://coveralls.io/github/cjrh/misu?branch=master

.. image:: https://img.shields.io/pypi/pyversions/misu.svg
    :target: https://pypi.python.org/pypi/misu

.. image:: https://img.shields.io/github/tag/cjrh/misu.svg
    :target: https://img.shields.io/github/tag/cjrh/misu.svg

.. image:: https://img.shields.io/badge/install-pip%20install%20misu-ff69b4.svg
    :target: https://img.shields.io/badge/install-pip%20install%20misu-ff69b4.svg

.. image:: https://img.shields.io/pypi/v/misu.svg
    :target: https://img.shields.io/pypi/v/misu.svg

.. image:: https://img.shields.io/badge/calver-YYYY.MM.MINOR-22bfda.svg
    :target: http://calver.org/

misu
====

``misu`` is short for "misura", which means **measurement** (in
Italian). ``misu`` is a package for doing calculations with in consistent
units of measurement.

Install
-------

On Windows, precompiled wheels are provided so all you have to do is
this:

.. code-block:: shell

    pip install misu

On Linux, you have to install from a source distribution (sdist). This is
also on PyPI, but you must already have Cython and numpy present in your
target environment. This is because they are required to build *misu*.
Thus, you need something like this on Linux:

.. code-block:: shell

    $ python3.7 -m venv venv
    $ source venv/bin/activate
    (venv) $ pip install Cython numpy
    (venv) $ pip install misu

    <lots of compiler output>

If you have have experience with making *manylinux* wheels for Linux, I
would love to get your help to make them for *misu* too!

Demo
----

Most of the time you will probably work with ``misu`` interactively, and
it will be most convenient to import the unit symbols into the global namespace:

.. code:: python

    import misu

    u = misu.UnitNamespace(context='all')
    misu.units_to_this_ns(u)

    mass = 100*kg
    print(mass >> lb)

The symbol ``kg`` defined in the UnitNamespace u is no known in global
namespace. We redefine the shift operator to perform inline conversions. The
code above produces:

::

    220.46226218487757

There are many units already defined (look at `unitdefs.json`), and it is easy
to add more. Here we convert the same quantity into ounces:

.. code:: python

    print(mass >> oz)

output:

::

    3571.4285714285716

What you see above would be useless on its own. What you really need is
to be able to perform consistent calculations with quantities expressed
in different, but compatible units:

.. code:: python

    mass = 10*kg + 20*lb
    print(mass)

output:

::

    19.07 kg

For addition and subtraction, ``misu`` will ensure that only consistent
units can be used. Multiplication and division will produce new units:

.. code:: python

    distance = 100*metres
    time = 9.2*seconds

    speed = distance / time
    print(speed)

output:

::

    10.87 m/s

As before, it is trivially easy to express that quantity in different
units of compatible dimensions:

.. code:: python

    print(speed >> km/hr)

output:

::

    39.130434782608695

Introduction
------------

``misu`` is a package of handling physical quantities with dimensions.
This means performing calculations with all the units being tracked
correctly. It is possible to add *kilograms per hour* to *ounces per
minute*, obtain the correct answer, and have that answer be reported in,
say, *pounds per week*.

``misu`` grew out of a personal need. I have used this code personally
in a (chemical) engineering context for well over a year now (at time of
writing, Feb 2015). Every feature has been added in response to a
personal need.

Features
^^^^^^^^

-  Speed optimized. ``misu`` is very fast! Heavy math code in Python
   will be around only 5X slower when used with ``misu``. This is much
   faster than other quantities packages for Python.

-  Written as a Cython extension module. Speed benefits carry over when
   using ``misu`` from your own Cython module (a ``.pxd`` is provided
   for linking).

-  When an operation involving incompatible units is attempted, an
   ``EIncompatibleUnits`` exception is raised, with a clear explanation
   message about which units were inconsistent.

-  Decorator for functions to enforce dimensions

.. code:: python

    @dimensions(x='Length', y='Mass')
    def f(x, y):
        return x/y

    f(2*m, 3*kg)         # Works
    f(200*feet, 3*tons)  # Works

    f(2*joules, 3*kelvin)  # raises AssertionError
    f(2*m, 3)              # raises AssertionError

-  Decorator for functions to convert the input quantities to a specified unit,
   perform the calculation only on the magnitude and add the output units to
   return a Quantity again.

    @calc_unitless([m/s, s], x=m, y=s)
    def f(x, y):
        return x/y, y

    print(f(2*m, 3*s))      # returns [0.667 m/s, 3 s]

    print(f(2*kg, 3*s))     # raises EIncompatibleUnits error.

-  An operator for easily stripping the units component to obtain a
   plain numerical value

.. code:: python

    mass = 100 * kg
    mass_lb = mass >> lb

    duty = 50 * MW
    duty_BTU_hr = duty >> BTU / hr

-  An enormous amount of redundancy in the naming of various units. This
   means that ``m``, ``metre``, ``metres``, ``METRE``, ``METRES`` will
   all work. The reason for this is that from my own experience, when
   working interactively (e.g. in the IPython Notebook) it can be very
   distracting to incorrectly guess the name for a particular unit, and
   have to look it up. ``ft``, ``foot`` and ``feet`` all work, ``m3``
   means ``m**3`` and so on.
-  You can specify a *reporting unit* for a dimension, meaning that you
   could have all lengths be reported in "feet" by default for example. You can
   specify a *reporting format* for a particular unit. This is realized by
   using the `Quantity.setRepresent` method. Its `convert_function` argument
   even allows you to report in another unit category, for example,
   automatically report all energies in Hz (by dividing by Planck's constant.)
-  As ``misu`` does not support quantities that have an offset wrt. SI units (the most common example are the temperature units Farenheit and Celsius), it ships with helper functions `misu.xx_val_from_yy(yy-magnitude)` to convert magnitudes between those units.
-  Misu ships with build in physical constants which are taken from scipy if installed. If scipy is not found, fallback values are used. See `misu.__init__.py` to see the main objects the library supplies.

There are other projects, why ``misu``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several units systems for Python, but the primary motivating
use-case is that ``misu`` is written as a Cython module and is by far
the fastest\* for managing units available in Python.

\*\ *Except for ``NumericalUnits``, which is a special case*

\*\*\ *I haven't actually checked that this statement is true for all of
them yet.*

General usage
-------------

For speed-critical code, the application of unit operations can still be too
slow. In these situations it is typical to first cast quantities into numerical
values (doubles, say), perform the speed-critical calculations (perhaps call
into a C-library), and then re-cast the result back into a quantity and return
that from a function. That's the intention behind the `@calc_unitless`
decorator explained above. This way you can still easily wrap
performance-critical calculations with robust unit-handling.

Inspiration
^^^^^^^^^^^

The inspiration for ``misu`` was
`Frink <http://futureboy.us/frinkdocs/>`__ by Alan Eliasen. It is
*wonderful*, but I need to work with units in the IPython Notebook, and
with all my other Python code.

There are a bunch of other similar projects. I have not used any of them
enough yet to provide a fair comparison:

-  `astropy.units <http://astropy.readthedocs.org/en/latest/units/>`__
-  `Buckingham <http://code.google.com/p/buckingham/>`__
-  `DimPy <http://www.inference.phy.cam.ac.uk/db410/>`__
-  `Magnitude <http://juanreyero.com/open/magnitude/>`__
-  `NumericalUnits <https://pypi.python.org/pypi/numericalunits>`__
-  `Pint <http://pint.readthedocs.org/>`__
-  `Python-quantities <https://pypi.python.org/pypi/quantities>`__
-  `Scalar <http://russp.us/scalar-guide.htm>`__
-  `Scientific.Physics.PhysicalQuantities <http://dirac.cnrs-orleans.fr/ScientificPython/ScientificPythonManual/Scientific.Physics.PhysicalQuantities-module.html>`__
-  `SciMath <http://scimath.readthedocs.org/en/latest/units/intro.html>`__
-  `sympy.physics.units <http://docs.sympy.org/dev/modules/physics/units.html>`__
-  `udunitspy <https://github.com/blazetopher/udunitspy>`__
-  `Units <https://bitbucket.org/adonohue/units/>`__
-  `Unum <https://bitbucket.org/kiv/unum/>`__


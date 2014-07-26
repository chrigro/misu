# Always prefer setuptools over distutils 
from setuptools import setup, find_packages 
from codecs import open # To use a consistent encoding
from os import path

# Get the long description from the relevant file 
here=path.abspath(path.dirname(__file__)) 
with open(path.join(here, 'README'), encoding='utf-8') as f:
    long_description=f.read()
    
setup(
    name='unity',
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version='1.0.0',
    description='Fast quantities',
    long_description=long_description,
    #this is the project's main homepage.
    url='https://github.com/cjrh/unity'
    # Author details
    author='Caleb Hattingh',
    author_email='caleb.hattingh@gmail.com',
    # Choose your license
    license='MIT',
    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        ],
    # What does your project relate to?
    keywords='math science engineering physics quantities units',
    packages=['unity']
    )
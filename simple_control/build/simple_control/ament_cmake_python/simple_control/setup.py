from setuptools import find_packages
from setuptools import setup

setup(
    name='simple_control',
    version='0.0.0',
    packages=find_packages(
        include=('simple_control', 'simple_control.*')),
)

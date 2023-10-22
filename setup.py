#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='DLAD_ex1',
    version='1.0',
    description='Exercise 1',
    install_requires=requirements,
    packages=find_packages()
)

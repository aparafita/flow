#!/usr/bin/env python

from distutils.core import setup
import setuptools

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name='flow',
    version='1.0',
    description='Flow models in PyTorch',
    author='Álvaro Parafita',
    author_email='parafita.alvaro@gmail.com',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=install_requires
)
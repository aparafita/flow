#!/usr/bin/env python

from setuptools import setup

version = '0.1.2'

long_description = """
# flow

This project implements basic Normalizing Flows in PyTorch 
and provides functionality for defining your own easily, 
following the conditioner-transformer architecture.

This is specially useful for lower-dimensional flows and for learning purposes.
Nevertheless, work is being done on extending its functionalities 
to also accomodate for higher dimensional flows.

Supports conditioning flows, meaning, learning probability distributions
conditioned by a given conditioning tensor. 
Specially useful for modelling causal mechanisms.

For more information, 
please look at our [Github page](https://github.com/aparafita/flow).
"""

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name='flow-torch',
    packages=['flow'],
    version=version,
    license='MIT',
    description='Normalizing Flow models in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Álvaro Parafita',
    author_email='parafita.alvaro@gmail.com',
    url='https://github.com/aparafita/flow',
    download_url=f'https://github.com/aparafita/flow/archive/v{version}.tar.gz',
    keywords=[
        'flow', 'density', 'estimation', 
        'sampling', 'probability', 'distribution'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
    ],
    install_requires=install_requires,
    include_package_data=True,
)

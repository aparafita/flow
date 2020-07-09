#!/usr/bin/env python

from distutils.core import setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name='flow-torch',
    packages=['flow'],
    version='0.1',
    license='MIT',
    description='Normalizing Flow models in PyTorch',
    long_description=readme,
    author='Álvaro Parafita',
    author_email='parafita.alvaro@gmail.com',
    url='https://github.com/aparafita/flow-torch',
    download_url='https://github.com/aparafita/flow-torch/archive/v0.1.tar.gz',
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
    install_requires=install_requires
)
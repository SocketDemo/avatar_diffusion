from setuptools import setup, find_packages
import os

setup(
    name='avatar_diffusion',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'diffusers==0.13.1',
        'flask==2.2.3',
        'numpy==1.23.5',
        'torch==1.13.1',
        'werkzeug==2.2.3',
        'requests5'
    ]
)
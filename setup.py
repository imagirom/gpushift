"""
gpushift - Differentiable Mean Shift on the GPU using PyKeOps and PyTorch.
"""

from setuptools import setup

setup(
    name='gpushift',
    version='0.1',
    packages=['gpushift'],
    url='',
    license='Apache Software License 2.0',
    author='Roman Remme',
    author_email='roman.remme@iwr.uni-heidelberg.de',
    description='Mean Shift on the GPU using PyKeOps and PyTorch',
    install_requires=[
        "torch>=1.0",
        "pykeops>=1.2"
    ]
)

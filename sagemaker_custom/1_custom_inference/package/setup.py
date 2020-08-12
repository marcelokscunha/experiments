# !pygmentize package/setup.py

from __future__ import absolute_import

from glob import glob
import os
from os.path import basename
from os.path import splitext

from setuptools import find_packages, setup

setup(
    name='custom_lightgbm_inference',
    version='0.1.0',
    description='Custom container serving package for SageMaker.',
    keywords="custom container serving package SageMaker",

    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    
    install_requires=[
        'sagemaker-inference==1.3.0',
        'multi-model-server==1.1.2'
    ],
    entry_points={"console_scripts": ["serve=custom_lightgbm_inference.my_serving:main"]},
)

from setuptools import setup

# trying to import the required torch package
try:
    import torch
except ImportError:
    raise Exception('qsketch requires PyTorch to be installed. aborting')

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Proceed to setup
setup(
    name='torchpercentile',
    version='0.2',
    description='A percentile implementation for pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/aliutkus/torchpercentile',
    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    keywords='percentile',
    install_requires=[
        'torchsearchsorted @ git+https://github.com/aliutkus/torchsearchsorted',
    ],

    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr')

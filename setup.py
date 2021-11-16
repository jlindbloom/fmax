from setuptools import setup, find_packages

setup(
    name='fmax',
    version='0.1.0',
    author='Jonathan Lindbloom',
    author_email='jonathan@lindbloom.com',
    license='LICENSE',
    packages=find_packages(),
    description='A Python package for fitting max/min time series forecasting models.',
    long_description=open('README.md').read(),
)
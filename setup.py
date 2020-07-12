from setuptools import setup, find_packages

setup(
    name='moe',  # Required
    version='0.0.0',  # Required
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'cytoolz',
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
    ],
)

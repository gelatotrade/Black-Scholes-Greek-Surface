#!/usr/bin/env python3
"""Setup script for Black-Scholes Greek Surface package."""

from setuptools import setup, find_packages
import os

# Read README for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='black-scholes-greek-surface',
    version='1.0.0',
    description='Interactive 3D visualizations of Black-Scholes option Greeks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/Black-Scholes-Greek-Surface',
    author='Black-Scholes Greek Surface Contributors',
    author_email='',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='black-scholes options greeks visualization finance derivatives trading',
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'plotly>=5.10.0',
    ],
    extras_require={
        'full': [
            'kaleido>=0.2.1',
            'jupyter>=1.0.0',
            'ipywidgets>=8.0.0',
            'streamlit>=1.20.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'mypy>=0.990',
        ],
    },
    entry_points={
        'console_scripts': [
            'greek-surface=main:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/Black-Scholes-Greek-Surface/issues',
        'Source': 'https://github.com/yourusername/Black-Scholes-Greek-Surface',
    },
)

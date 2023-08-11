from setuptools import find_packages
from setuptools import setup

name = 'torchrua'

setup(
    name=name,
    version='0.5.0.a',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/torchrua',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='Manipulate tensors with PackedSequence and CattedSequence',
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'einops',
    ],
)

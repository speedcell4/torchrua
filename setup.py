from setuptools import setup, find_packages

name = 'torchrua'

setup(
    name=name,
    version='0.4.0',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/torchrua',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='Manipulate tensors with PackedSequence and CattedSequence',
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'einops',
    ],
    extras_require={
        'dev': [
            'pytest',
            'hypothesis',
            'torch_scatter',
        ],
        'benchmark': [
            'aku',
            'tqdm',
            'torch_scatter',
        ]
    }
)

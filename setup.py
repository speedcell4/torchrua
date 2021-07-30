from setuptools import setup, find_packages

name = 'torchrua'

setup(
    name=name,
    version='0.3.0',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/torchrua',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='Manipulate tensor batch with PackedSequence and CattedSequence',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'einops',
    ],
    extras_require={
        'dev': [
            'pytest',
            'hypothesis',
        ],
        'benchmark': [
            'aku',
            'tqdm',
        ]
    }
)

from pathlib import Path

from setuptools import find_packages, setup

name = 'torchrua'

root_dir = Path(__file__).parent.resolve()
with (root_dir / 'requirements.txt').open(mode='r', encoding='utf-8') as fp:
    install_requires = [install_require.strip() for install_require in fp]

setup(
    name=name,
    version='0.5.1',
    packages=[package for package in find_packages() if package.startswith(name)],
    url='https://github.com/speedcell4/torchrua',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='rua your variable-length tensors!',
    python_requires='>=3.8',
    install_requires=install_requires,
)

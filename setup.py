from setuptools import setup, find_packages
from codecs import open
from os import path

from src.version import __version__

here = path.abspath(path.dirname(__file__))
try:
    # Get the long description from the README file
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [line.strip() for line in f]

setup(
    name='text-rsisr',
    version=__version__,
    description='Real World Text Images Super Resolution',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Dmytro-Shvetsov/text-rsisr.git',
    author='Dmytro Shvetsov',
    author_email='shvetsovdi2@gmail.com',
    packages=find_packages(exclude=['data', 'pretrained_models']),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.6',
)
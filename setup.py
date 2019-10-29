
# Imports
from setuptools import setup, find_packages

# Installation setup
setup(
    name='EchoTorch',
    version='0.1.2',
    description="A Python toolkit for Reservoir Computing.",
    long_description="A Python toolkit for Reservoir Computing, Echo State Network and Conceptor experimentation based on pyTorch.",
    author='Nils Schaetti',
    author_email='nils.schaetti@unine.ch',
    license='GPLv3',
    packages=find_packages(),
    install_requires=[
      'torch==1.3.0',
      'torchvision==0.4.1'
    ],
    zip_safe=False
)


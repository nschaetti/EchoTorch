
# Imports
from setuptools import setup, find_packages

# Installation setup
setup(
    name='EchoTorch-PyTorch2Build',
    version='1.0.0-pre',
    description="A Python toolkit for Reservoir Computing",
    long_description="A Python toolkit for Reservoir Computing, Echo State Network and Conceptor experimentation "
                     "based on pyTorch"
                     "Updated version of the original work by Nils Schaetti",
    author='Raamesh Balabhadrapatruni',
    author_email='raameshb@proton.me',
    license='GPLv3',
    packages=find_packages(),
    zip_safe=False,
    download_url = 'https://github.com/RaameshB/EchoTorch/archive/refs/tags/v1.0.0-pre.tar.gz',
    install_requires = [
             'future',
             'numpy==2.1.2',
             'scipy==1.14',
             'matplotlib',
             'torch==2.5.1',
             'torchvision',
             'networkx',
             'tqdm'
    ]
)


# Imports
from setuptools import setup, find_packages

# Installation setup
setup(
    name='EchoTorch',
    version='1.0.0-pre',
    description="A Python toolkit for Reservoir Computing",
    long_description="A Python toolkit for Reservoir Computing, Echo State Network and Conceptor experimentation "
                     "based on pyTorch"
                     "Updated version of the original work by Nils Schaetti",
    author='Nils Schaetti',
    author_email='nils.schaetti@unige.ch',
    license='GPLv3',
    python_requires='>=3.11,<3.13',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    packages=find_packages(),
    zip_safe=False,
    download_url = 'https://github.com/nschaetti/EchoTorch/archive/refs/tags/v1.0.0-pre.tar.gz',
    install_requires = [
             'future',
             'numpy==2.1.2',
             'scipy==1.14',
             'matplotlib',
             'torch==2.8.0',
             'torchvision==0.23.0',
             'scikit-learn',
             'networkx',
             'tqdm'
    ]
)

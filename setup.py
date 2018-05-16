from setuptools import setup, find_packages

setup(name='EchoTorch',
      version='0.1.1',
      description="A Python toolkit for Reservoir Computing.",
      long_description="A Python toolkit for Reservoir Computing and Echo State Network experimentation based on pyTorch.",
      author='Nils Schaetti',
      author_email='nils.schaetti@unine.ch',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
          'torch',
          'numpy',
          'torchvision'
      ],
      zip_safe=False
      )


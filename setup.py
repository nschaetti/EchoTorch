from setuptools import setup

setup(name='EchoTorch',
      version='0.1',
      description="A Python toolkit for Reservoir Computing.",
      long_description="A Python toolkit for Reservoir Computing and Echo State Network experimentation based on pyTorch.",
      author='Nils Schaetti',
      author_email='nils.schaetti@unine.ch',
      license='GPLv3',
      packages=['dataset', 'nn', 'tools'],
      zip_safe=False,
      install_requires=[
          'torch',
          'Sphinx',
          'numpy'
          ],
      )

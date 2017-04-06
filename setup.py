from setuptools import setup

setup(name='EchoTorch',
      version='0.1',
      description="A Python toolkit for Reservoir Computing.",
      long_description="A Python toolkit for Reservoir Computing and Echo State Network experimentation based on pyTorch.",
      author='Nils Schaetti',
      author_email='nils.schaetti@unine.ch',
      license='GPLv3',
      packages=['echotorch'],
      zip_safe=False,
      dependency_links=[
            'http://download.pytorch.org/whl/torch-0.1.10.post1-cp35-cp35m-macosx_10_6_x86_64.whl'
      ]
      )


import sys
import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

CURRENT_PYTHON = sys.version_info[:2]
MIN_PYTHON = (3, 6)

if CURRENT_PYTHON < MIN_PYTHON:
  sys.stderr.write("""
      ============================
      Unsupported Python Version
      ============================

      Python {}.{} is unsupported. Please use a version newer than Python {}.{}.
  """.format(*CURRENT_PYTHON, *MIN_PYTHON))
  sys.exit(1)

with open('requirements.txt', 'r') as f:
  install_requires = f.readlines()

with open('requirements_extra.txt', 'r') as f:
  extra_install_requires = f.readlines()

if os.path.isfile('VERSION'):
  with open('VERSION') as f:
    VERSION = f.read()
else:
  VERSION = os.environ.get('TRAVIS_PULL_REQUEST_BRANCH') or \
            os.environ.get('TRAVIS_BRANCH') or \
            '0.0.dev0'

with open('README.rst') as f:
  README = f.read()


class PyTest(TestCommand):
  def initialize_options(self):
    TestCommand.initialize_options(self)
    self.pytest_args = ""

  def run_tests(self):
    import shlex
    import pytest
    errno = pytest.main(shlex.split(self.pytest_args))
    sys.exit(errno)


setup(name='torchrl',
      description='Reinforcement Learning for PyTorch',
      long_description=README,
      long_description_content_type='text/x-rst',
      version=VERSION,
      url='https://torchrl.sanyamkapoor.com',
      author='Sanyam Kapoor',
      license='Apache License 2.0',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages(exclude=[
          'tests',
          'tests.*',
          'experiments',
          'experiments.*'
      ]),
      tests_require=[
          'pylint>=2.2',
          'pytest>=4.2'
      ],
      install_requires=install_requires,
      extras_require={
          'docs': [
              'sphinx>=1.8',
              'sphinx-rtd-theme>=0.4',
              'sphinxcontrib-napoleon>=0.7',
              'm2r>=0.2.0',
              'sphinxcontrib-programoutput>=0.11',
          ],
          'extra': extra_install_requires,
      },
      entry_points={
          'console_scripts': [
              'torchrl=torchrl.cli.boot:main',
          ]
      },
      cmdclass={"test": PyTest})

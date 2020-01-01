import os
from setuptools import setup, find_packages


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
          'experiments',
          'experiments.*'
      ]),
      python_requires='>=3.6',
      install_requires=install_requires,
      extras_require={
          'test': [
              'pylint>=2.2',
              'pytest>=4.2',
          ],
          'docs': [
              'sphinx>=1.8',
              'sphinx-rtd-theme>=0.4',
              'sphinxcontrib-napoleon>=0.7',
              'm2r>=0.2.0',
              'sphinxcontrib-programoutput>=0.11',
          ],
          'extra': extra_install_requires,
      }
     )

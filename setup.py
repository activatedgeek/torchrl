import sys
from setuptools import setup, find_packages

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

with open('requirements_dev.txt', 'r') as f:
  dev_install_requires = f.readlines()

with open('VERSION') as f:
    VERSION = f.read().strip()

with open('README.rst') as f:
    README = f.read()

setup(name='torchrl',
      description='Reinforcement Learning for PyTorch',
      long_description=README,
      long_description_content_type='text/x-rst',
      version=VERSION,
      url='https://www.github.com/activatedgeek/torchrl',
      author='Sanyam Kapoor',
      license='Apache License 2.0',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages(),
      install_requires=install_requires,
      extras_require={
        'dev': dev_install_requires,
      },
      entry_points={
        'console_scripts': [
          'torchrl=torchrl.cli:main'
        ]
      }
)

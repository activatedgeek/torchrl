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

with open('VERSION') as f:
    VERSION = f.read().strip()

setup(name='rl_baselines',
      version=VERSION,
      url='https://www.github.com/activatedgeek/rl_baselines',
      author='Sanyam Kapoor',
      license='MIT',
      classifiers=[
          'Programming Language :: Python :: 3.6',
      ],
      packages=find_packages(),
      install_requires=install_requires)

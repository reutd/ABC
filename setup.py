from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ABC',
    version='0.1',
    packages=['ABC'],
    url='',
    license='MIT License',
    author='Reut Danino',
    description='ABC (Autoencoder-based Batch Correction) is a semi-supervised deep learning architecture for integrating single cell sequencing datasets',
    install_requires=required,
)

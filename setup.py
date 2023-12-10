from setuptools import setup, find_packages

setup(
    name='smtlayer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',  # Add other dependencies here
        'torchvision',
        'tqdm',
        'numpy',
        'z3-solver',
        'click'
    ],
)
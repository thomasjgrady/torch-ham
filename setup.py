from setuptools import setup

setup(
    name='torch_ham',
    version='0.0.1',
    description='Hierarchical associative memory in PyTorch',
    url='https://github.com/thomasjgrady/torch-ham',
    author='Thomas Grady',
    author_email='tgrady@gatech.edu',
    license='Apache 2.0',
    packages=['torch_ham'],
    install_requires=[
        'torch'
    ]
)
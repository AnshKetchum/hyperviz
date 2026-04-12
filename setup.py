from setuptools import setup, find_packages

setup(
    name="transformer-visualization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "matplotlib",
    ],
)

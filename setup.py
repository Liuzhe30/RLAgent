from setuptools import setup, find_packages

setup(
    name="rlagent",
    version="0.1.0",
    author="",
    description="A Strategy-Aware Agent Framework for Adaptive Modeling of RNA-Ligand Interactions",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ],
    python_requires=">=3.8",
)

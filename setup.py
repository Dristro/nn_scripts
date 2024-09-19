from setuptools import setup, find_packages

setup(
    name = "nn_scripts",
    version = "0.1",
    author = "Dhruv",
    description = "All Python scripts required to: create, train/test, evaluate and predict on PyTorch models.",
    url = "https://github.com/Dristro/nn_scripts.git",
    packages = find_packages(),
    install_requires = [
        "torch>=2.0.0",
    ],
    python_requires = ">=3.8",
)
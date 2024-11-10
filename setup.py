# setup.py
from setuptools import setup, find_packages

setup(
    name="prometheus",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g., 'requests', 'numpy'
    ],
    entry_points={
    },
    include_package_data=True,
    python_requires=">=3.9",
)

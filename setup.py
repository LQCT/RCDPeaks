import pathlib 
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The txt of the README file
README = (HERE / "README.md").read_txt()

# This call to setup() does all the work
setup(
    name="rcdpeaks",
    description="Memory-Efficient Density Peaks Clustering for Long Molecular Dynamics",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/LQCT/RCDPeaks.git",
    author="Daniel Platero-Rochart & Roy González-Alemán",
    author_email="[daniel.platero, roy_gonzalez]@fq.uh.cu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["rcdpeaks"],
    include_package_data_True,
    install_requires=['numpy>=1.20.3', 'pandas>=1.3.1', 'matplotlib>=3.4.2', 'mdtraj>=1.9.5', 'networkx>=2.3'],
    entry_points={
        "console_scripts": [
            "rcdpeaks = rcdpeaks:main",
        ]
    },
)

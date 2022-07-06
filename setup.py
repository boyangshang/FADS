import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="FADS",
    version="1.0.1",
    description="Fast Diversity Subsampling from a Data Set",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/boyangshang/FADS",
    author="Boyang Shang, Daniel Apley, Sanjay Mehrotra",
    author_email="boyangshang2015@u.northwestern.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(include=['FADS', 'FADS.*'],exclude=['tests']),
    include_package_data=True,
    install_requires=["numpy", "scikit-learn"],
    tests_require=['scipy','pytest'],
)

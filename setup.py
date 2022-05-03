import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="FADS",
    version="1.0.2",
    description="Fast Diversity Subsampling from a Data Set",
    long_description_content_type="text/markdown",
    long_description=README,
    url="https://github.com/boyangshang/FADS",
    author="Boyang Shang",
    author_email="boyangshang2015@u.northwestern.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["FADS"],
    include_package_data=True,
    install_requires=["numpy", "scikit-learn"],
)

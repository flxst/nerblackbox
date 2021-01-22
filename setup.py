from setuptools import setup, find_packages
import pathlib
import os.path
import codecs

BASE_DIR = pathlib.Path(__file__).parent.resolve()


def readme():
    with open(f"{BASE_DIR}/README.rst") as f:
        return f.read()


def requirements():
    with open(f"{BASE_DIR}/requirements.txt") as f:
        return f.read().splitlines()


def requirements_dev():
    with open(f"{BASE_DIR}/requirements_dev.txt") as f:
        return f.read().splitlines()


def get_package_version(rel_path):
    def read(_rel_path):
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, _rel_path), "r") as fp:
            return fp.read()

    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="nerblackbox",
    version=get_package_version("nerblackbox/__about__.py"),
    author="Felix Stollenwerk",
    author_email="felix.stollenwerk@arbetsformedlingen.se",
    description="fine-tune transformer-based models for named entity recognition",
    long_description=readme(),
    long_description_content_type="text/x-rst",
    keywords=[
        "NLP",
        "NER",
        "named entity recognition",
        "BERT",
        "transformer",
        "pytorch",
    ],
    url="https://pypi.org/project/nerblackbox",
    license="Apache 2.0",
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=requirements(),
    extras_require={
        "dev": requirements_dev(),
    },
    python_requires=">=3.6",
    entry_points="""
            [console_scripts]
            nerbb=nerblackbox.cli:nerbb
        """,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
)


from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='nerblackbox',
    version='0.0.4',
    author='Felix Stollenwerk',
    author_email='felix.stollenwerk@arbetsformedlingen.se',
    description='fine-tune pretrained transformer-based models for named entity recognition',
    long_description=readme(),
    keywords='NLP BERT NER transformer named entity recognition pytorch',
    url='',
    license='Apache',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=requirements(),
    python_requires=">=3.6.0",
    entry_points='''
            [console_scripts]
            nerbb=nerblackbox.cli:main
        ''',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Unix',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
)

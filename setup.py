
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='ner_black_box',
    version='0.0.1',
    author='Felix Stollenwerk',
    author_email='felix.stollenwerk@arbetsformedlingen.se',
    description='fine-tune pretrained transformer models on the named entity recognition downstream task',
    long_description=readme(),
    keywords='NLP BERT NER transformer named entity recognition pytorch',
    url='',
    license='Apache',
    packages=['ner_black_box'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=requirements(),
    python_requires=">=3.6.0",
    entry_points={
        'console_scripts': ['nerbb=ner_black_box.ner_black_box_cli:main'],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.6',
        'Operating System :: Unix',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
)

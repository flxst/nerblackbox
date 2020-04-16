
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='ner_black_box',
    version='0.1',
    description='fine-tune pretrained transformer models on the named entity recognition downstream task',
    long_description=readme(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='transformers bert nlp named entity recognition',
    url='',
    author='Felix Stollenwerk',
    author_email='felix.stollenwerk@arbetsformedlingen.se',
    license='MIT',
    packages=['ner_black_box'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=requirements(),
    entry_points={
        'console_scripts': ['nerbb=ner_black_box.ner_black_box_cli:main'],
    },
    include_package_data=True,
    zip_safe=False,
)

"""Package setup script for Yogurt."""

from setuptools import find_packages
from setuptools import setup


with open('README.md', encoding='utf-8') as file:
    readme_text = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = [
        line
        for rawline in file.readlines()
        if (line := rawline.strip()) and not line.startswith('#')
    ]


setup(
    name='yogurt',
    version='0.0.1',
    description='Context Aware YO Letter Restoration in Russian Texts ',
    long_description=readme_text,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['nlp', 'deeplearning', 'russian', 'yo', 'yoficator', 'yofication'],
    packages=find_packages(exclude=['scripts', 'tests']),
    python_requires='>=3.9',
    install_requires=requirements,
)

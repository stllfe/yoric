"""Package setup script for Yogurt."""

from setuptools import setup


NAME = 'yoric'

with open('README.md', encoding='utf-8') as file:
    readme_text = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = [
        line
        for rawline in file.readlines()
        if (line := rawline.strip()) and not line.startswith('#')
    ]


setup(
    name=NAME,
    version='0.1.0',
    description='Context Aware YO Letter Restoration in Russian Texts.',
    long_description=readme_text,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['nlp', 'deeplearning', 'russian', 'yo', 'yoficator', 'yofication'],
    packages=[NAME],
    python_requires='>=3.9',
    install_requires=requirements,
)

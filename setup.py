from setuptools import setup, find_packages

setup(
    name='slovakey',
    version='1.0',
    author='Lívia Kelebercová',
    author_email='livia.kelebercova@gmail.com',
    description='A package for keyword extraction in Slovak text',
    packages=find_packages(),
    install_requires=[
        'sentence-transformers==2.1.0',
        'numpy==1.21.5',
        'stanza==1.3.0',
        'scikit-learn==0.24.2',
        'nltk==3.6.5',
        'torch==1.10.0'
    ],
)

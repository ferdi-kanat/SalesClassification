from setuptools import setup, find_packages

setup(
    name="sales-classification",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'seaborn>=0.11.0',
        'matplotlib>=3.4.0',
        'deap>=1.3.1',
    ],
    python_requires='>=3.8',
    author="Ferdi Kanat",
    author_email="200101038@ogrenci.yalova.edu.tr",
    description="A machine learning pipeline for sales profit prediction",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ferdi-kanat/SalesClassification",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

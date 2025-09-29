"""
BERTopic Enhanced - Production Ready Knowledge Graph Topic Modeling

Setup script for installing the package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = (
        "BERTopic Enhanced - Production Ready Knowledge Graph Topic Modeling"
    )

setup(
    name="bertopic-enhanced",
    version="2.0.0",
    description="Enhanced BERTopic with document preprocessing, knowledge graphs, and incremental indexing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BERTopic Community",
    author_email="",
    url="https://github.com/your-repo/bertopic-enhanced",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="topic-modeling bertopic nlp machine-learning clustering knowledge-graph",
    python_requires=">=3.8",
    install_requires=[
        "bertopic>=0.16.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.0.0",
        "seaborn>=0.11.0",
        "wordcloud>=1.8.0",
        "umap-learn>=0.5.0",
        "hdbscan>=0.8.0",
        "sentence-transformers>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "flake8>=4.0.0",
        ],
        "preprocessing": [
            "langdetect>=1.0.0",
            "spacy>=3.0.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "networkx>=2.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "bertopic-enhanced=bertopic.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

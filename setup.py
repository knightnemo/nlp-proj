from setuptools import setup, find_packages

setup(
    name="llm-simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A text-based world simulator using language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-simulator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 
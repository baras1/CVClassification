from setuptools import setup, find_packages

# Load the README.md content to display it on PyPI or other package managers
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="CVClassification",  # Name of your project
    version="0.1.0",  # Initial version number
    author="Bara Anugrah Salim",
    author_email="baras1@uw.edu",
    description="A project that showcases image classification using PyTorch and NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baras1/CVClassification",  # Link to your project
    packages=find_packages(),  # Automatically find any modules in your project
    install_requires=[  # Dependencies listed here should match what's in your requirements.txt
        'matplotlib==3.9.2',
        'numpy==2.1.1',
        'pillow==10.4.0',
        'torch==2.4.1',
    ],
    classifiers=[  # Metadata about your project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # Minimum Python version requirement
    entry_points={  # This allows users to run "cvclassify" in the terminal
        'console_scripts': [
            'cvclassify=main:main',  # Maps the "cvclassify" command to the main() function in main.py
        ],
    },
)

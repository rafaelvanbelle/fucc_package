import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fucc", 
    version="0.0.2",
    author="Rafael Van Belle",
    author_email="rafael@gmail.com",
    description="Fraudulent Use of Credit Card",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rafaelvanbelle/fucc_package",
    packages=setuptools.find_packages(),
    install_requires=[
          'scikit-learn',
      ],
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="salmon",
    version="0.0.0",
    classifiers=[
		"Development Status :: 4 - Beta",
		"Intended Audience :: Developers",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3.6",
    ],
    description="🐟",
    long_description=long_description,
    long_description_content_type="text/markdown",
	author="Maciej Kilian",
    url="https://github.com/iejMac/salmon",
    license="MIT",
    packages=find_packages(),
    keywords=["machine learning"],
)

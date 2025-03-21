from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="meegnet",
    packages=find_packages(),
    version="0.2.5",
    description="A set of MEG-optimized ANNs and visualization tools.",
    author="Arthur Dehgan",
    license="MIT",
    install_requires=requirements,  # Use requirements.txt contents
)

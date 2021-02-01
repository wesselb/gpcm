from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "matplotlib",
    "plum-dispatch",
    "backends",
    "backends-matrix>=0.3.3",
    "varz",
    "wbml",
    "torch",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)

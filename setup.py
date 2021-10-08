from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "matplotlib",
    "plum-dispatch>=1",
    "backends>=1.4.9",
    "backends-matrix>=1.1.5",
    "stheno>=1.1.7",
    "mlkernels>=0.3.2",
    "varz>=0.7.4",
    "wbml>=0.3",
    "probmods>=0.3.0",
    "jax",
    "jaxlib",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)

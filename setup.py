from setuptools import find_packages, setup

setup(
    name="SpatialFormer",
    version="1.5.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    license="MIT",
    packages=find_packages(),
)

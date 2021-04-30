from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Makhanko_Maksim",
    test_suite='tests'
    install_requires=[
        "Flask==1.0.2",
        "marshmallow_dataclass==8.4.1",
        "numpy==1.15.1",
        "click==7.1.2",
        "requests==2.19.1",
        "pandas==0.23.4",
        "dataclasses==0.8",
        "PyYAML==5.4.1",
        "scikit_learn==0.24.2",s
    ],
    license="MIT",
)

# src/setup.py
from setuptools import setup, find_packages

setup(
    name="vehicle_insurance_prediction",
    version="0.0.1",
    author="Ravi",
    author_email="ravipanavir@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # IMPORTANT: do NOT require numpy/pandas/scikit-learn here.
    install_requires=[
        "dill",
        "PyYAML",
        "pymongo",
        "boto3",
        "joblib",
        # do not include numpy/pandas/scikit-learn/matplotlib here
    ],
)
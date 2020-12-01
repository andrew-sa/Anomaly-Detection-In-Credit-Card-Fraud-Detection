from setuptools import setup, find_packages

setup(
    name='anomaly-detection',
    version='0.1.0',
    description='Anomaly detection in credit card fraud detection',
    author='Andrea Mogavero',
    author_email='andreamogavero.sa@gmail.com',
    url='https://github.com/andrew-sa/Anomaly-Detection-In-Credit-Card-Fraud-Detection',
    package_dir={'': 'src'},
    packages=find_packages(where='src')
)
from setuptools import setup, find_packages

setup(
    name="etsi_watchdog",
    version="0.1.0",
    packages=find_packages(),  
    install_requires=[
        "Flask>=2.3.0",
        "Flask-CORS>=4.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "PyYAML>=6.0",
    ],
    include_package_data=True,
)

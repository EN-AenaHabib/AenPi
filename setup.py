from setuptools import setup, find_packages

setup(
    name="AenPi",
    version="0.1.1",
    description="Lightweight Urdu NLP Library",
    packages=[
        "AenPi",
        "AenPi.urdu",
    ],
    package_data={
        "AenPi.urdu": ["*.py"],
    },
    install_requires=[
        "psutil>=5.9.0",
    ],
    python_requires=">=3.7",
)

from setuptools import setup, find_packages

setup(
    name="AenPi",
    version="0.1.0",
    description="Lightweight Urdu NLP Library",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "psutil>=5.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: Urdu",
        "Topic :: Text Processing :: Linguistic",
    ],
)

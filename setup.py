from setuptools import setup, find_packages
setup(
    name="AenPi",
    version="0.1.2",
    description="Lightweight Urdu NLP Library",
    packages=[
        "AenPi",
        "AenPi.urdu",
    ],
    package_data={
        "AenPi.urdu": ["*.py", "*.pkl.gz", "*.joblib"],  # ← added *.joblib
    },
    include_package_data=True,
    install_requires=[
        "psutil>=5.9.0",
        "sklearn-crfsuite",
    ],
    python_requires=">=3.7",
)

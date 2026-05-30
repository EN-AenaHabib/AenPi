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
        "AenPi.urdu": ["*.py", "aenpi_pos_crf.pkl.gz"],  # merged into one key
    },
    include_package_data=True,
    install_requires=[
        "psutil>=5.9.0",
        "sklearn-crfsuite",  # ← added
    ],
    python_requires=">=3.7",
)

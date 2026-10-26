from setuptools import setup, find_packages

setup(
    name="dataset_generator",
    version="0.1.0",
    description="Synthetic dataset generator for the Space Debris challenge",
    author="Magdalena Machacka",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib"
    ],
)

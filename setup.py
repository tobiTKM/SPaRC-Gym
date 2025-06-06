from setuptools import setup, find_packages
import pathlib

# read the long description from README.md
this_dir = pathlib.Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="Gym-Env-SPaRC",              
    version="0.1.0",                       
    author="Tobias Mark",
    author_email="tobi09.mark@gmx.net",
    description="A Gymnasium environment for SPaRC puzzles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tobiTKM/Gym-Environment_for_SPaRC",  
    license="MIT",
    packages=find_packages(include=["gymnasium_env_for_SPaRC*",]),
    install_requires=[
        "gymnasium>=0.28.1",
        "numpy>=1.26.4",
        "pygame>=2.2.0",
        "pyyaml>=5.1",
        "pandas>=2.2.1",
    ],
    include_package_data=True,
    entry_points={
        "gymnasium.envs": [
            "env-SPaRC-v0 = gymnasium_env_for_SPaRC.gym_env_for_SPaRC:GymEnvSPaRC"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
import codecs
import os.path

from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "rt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="Kepler-RaDex2Pix",
    version=get_version("raDec2Pix/__init__.py"),
    author="Steve Bryson",
    author_email="steve.bryson@nasa.gov",
    url="https://github.com/stevepur/Kepler-RaDex2Pix",
    license="GPLv3",
    packages=["raDec2Pix"],
    package_dir={"raDec2Pix": "raDec2Pix"},
    package_data={"raDec2Pix": [
        "data/de421.bsp",
        "data/naif0012.tls",
        "data/spk_2018127000000_2018128182705_kplr.bsp",
        "data/geometryConstants.txt",
        "data/geometryUncertainty.txt",
        "data/pointingDeclinations.txt",
        "data/pointingMjds.txt",
        "data/pointingRas.txt",
        "data/pointingRolls.txt",
        "data/pointingSegmentStartMjds.txt",
        "data/rollTimeFovCenterDecs.txt",
        "data/rollTimeFovCenterRas.txt",
        "data/rollTimeFovCenterRolls.txt",
        "data/rollTimeMjds.txt",
        "data/rollTimeRollOffsets.txt",
        "data/rollTimeSeasons.txt",
    ]},
    install_requires=requirements,
)

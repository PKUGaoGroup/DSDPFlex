from setuptools import setup, find_packages

setup(
    name="DSDPFlex_pyTools",
    version="0.1",
    author="CW Dong",
    url="https://github.com/PKUGaoGroup/DSDPFlex",
    description="A data processing tool for DSDP",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "DSDPflex-py = DSDP_Tools.DSDPFlex:main",
        ],
    },
)

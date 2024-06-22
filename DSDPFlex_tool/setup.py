from setuptools import setup, find_packages

setup(
    name="DSDP_Tools",
    version="0.1",
    author="CW Dong",
    url="https://github.com/PKUGaoGroup/DSDP",
    description="A data processing tool for DSDP",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "DSDPFlex = DSDP_Tools.DSDPFlex:main",
        ],
    },
)

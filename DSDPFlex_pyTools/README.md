# DSDP_Tools

## Installation

build a conda env
```
conda create -n DSDPFlex
conda activate DSDPFlex
```

make sure DSDP(flex) and GNINA are in the system PATH. The current version uses `DSDPflex` as the binary name of the original DSDP(flex)

run this to install
```
python ./setup.py install
```

then can run docking with
```
DSDPFlex --ligand ... --protein ... --rescore gnina
```

NOTE: `DSDPFlex` is the command for calling this script

see more with 
`DSDPFlex --help`
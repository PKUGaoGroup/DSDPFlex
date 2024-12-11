# DSDPFlex: Flexible-Receptor Docking with GPU Acceleration

DSDPFlex is a GPU-accelerated flexible-receptor docking program derived from [DSDP (**D**eep **S**ite and **D**ocking **P**ose)](https://github.com/PKUGaoGroup/DSDP). Similar to [AutoDock Vina](https://vina.scripps.edu/), it supports selective side-chain flexibility during docking. A flexible docking process is typically completed in ~1 s with DSDPFlex. Learn more in our paper (DOI: [10.1021/acs.jcim.4c01715](https://doi.org/10.1021/acs.jcim.4c01715) )

## Installation

DSDPFlex runs on a Linux machine (tested on Ubuntu 22.04 and 20.04).

A GPU with CUDA is required. NVCC is used for compilation. Please install [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit) and make sure it is in the system PATH variable. (check with `nvcc --version`)

> [!NOTE]
> The CUDA version needs to be compatible with the GPU architecture and gcc/g++ version.
> 
> An example version we used is `cuda_11.6` with `gcc_9.4.0` (tested on NVIDIA RTX 3070Ti & RTX A4000). If an older GPU (former to the GTX/RTX Turing) is used on your computer, the option `-arch=sm_70` in `Makefile` needs to be modified to a compatible one.

Clone this repository.

```bash
git clone https://github.com/PKUGaoGroup/DSDPFlex.git
cd DSDPFlex
```

Install DSDPFlex. Suppose you are at the root of this repository,

```bash
cd DSDPFlex_v0.2
make clean && make
cp DSDPflex ../bin
cd ..
```

A binary named `DSDPflex` will be compiled. You can add it to the PATH variable by adding this line to `~/.bashrc` (if using Bash)
```bash
export PATH=/path/to/DSDPFlex/bin:$PATH
```

so that `DSDPflex` can be directly called from the command line.

## Using DSDPFlex

### Flexible docking with DSDPFlex.

(If the flexible receptor parts are not needed, [DSDP](https://github.com/PKUGaoGroup/DSDP) will work better.) 

For a ligand-receptor pair:

```bash
./DSDPflex --ligand ligand.pdbqt \
--protein receptor_rigid.pdbqt \
--flex receptor_flex.pdbqt \
--box_min x y z \
--box_max x y z \
--exhaustiveness 384 --search_depth 40 --top_n 1 \
--out ligand_out.pdbqt \
--out_flex flex_out.pdbqt \
--log dsdp_out.log
```

**Input:** DSDPFlex supports `.pdbqt` formatted input files, which can be prepared with [AutoDockTools](https://autodocksuite.scripps.edu/adt/).

- `--ligand` ligand file name (required)
- `--protein` protein rigid part (required)
- `--flex` protein flexible part file (required for flexible docking)

**Output:** (outputs will appear in the current work dir if path not specified)

- `--out` the output file of ligand poses (default=DSDP_out.pdbqt)
- `--out_flex` the output file of flexible side chain poses (default=DSDP_out_flex.pdbqt)
- `--log` the log file name (default=DSDP_out.log)
- `--top_n` the top-N ranking results will be exported (defualt=10)
  
**Search space:** The search space information needs to be provided. The *search box* specifies the (known) binding site. The *ligand box* is used to restrict ligand translation, which can be a smaller box. 

- `--box_min` x y z minima of the search box (in Angstrom)
- `--box_max` x y z maxima of the search box
- `--ligbox_min` x y z minima of the ligand box 
- `--ligbox_max` x y z maxima of the ligand box 

**Search settings:** Can be manually adjusted. The default settings generally work well.

- `--exhaustiveness` the number of sampling threads, default=384
- `--search_depth` the number of sampling iterations of each thread, default=40

Use `--help` to see more details.

### Using the ligand batch mode

A batch mode is designed for docking multiple ligands to a single receptor (e.g. in a virtual screening scenario).

```
./DSDPflex --ligand_batch batch_list --protein receptor_rigid.pdbqt --flex receptor_flex.pdbqt ...
```

The `batch_list` should be a text file, like
```
ligand_0.pdbqt out_0.pdbqt out_flex_0.pdbqt
ligand_1.pdbqt out_1.pdbqt out_flex_1.pdbqt
...
```
It will replace the `--ligand`, `--out`, and `--out_flex` options. Each file name should be a valid file path. The protein files (`--protein` & `--flex`) still need to be specified.

## Using DSDPFlex_pyTools

We provide a Python interface for calling DSDPFlex and related tools. It is recommended to run **rescoring** with this DSDPFlex_pyTools. More functions can be implemented in the future.

We recommend setting up the Python environment with conda.

```
conda create -n DSDPFlex_py
conda activate DSDPFlex_py
```

Then you can install the DSDPFlex_pyTools package by the following command:

```
cd ./DSDPFlex_pyTools
python ./setup.py install
```

Then the command `DSDPflex-py` will be available in the current python environment.

### Rescoring with GNINA

Before using `DSDPflex-py`, ensure that `DSDPflex` is in your system path (see Installation). To use GNINA, please install [GNINA](https://github.com/gnina/gnina) and add it to the PATH variable.

To use DSDPflex-py:
```bash
DSDPflex-py --ligand ... --protein ... --rescore gnina
```

(the other options are the same as `DSDPflex`.)  DSDPflex-py will call `gnina` for rescoring after docking by `DSDPflex`. By default, top-20 poses are rescored, and the output files will be named `*_rescored.pdbqt`

Other rescoring methods might be implemented in the future.

## Run DSDPFlex on APOBind core

The 229 systems within APOBind dataset, used for evaluation in the paper, are provided in `test/apobind_prepared`. 

To run this test:

```bash
cd ./test
sh ./run_apobind.sh
```

DSDPFlex will perform docking on all systems. We used [sPyRMSD](https://github.com/RMeli/spyrmsd) to calculate RMSDs in the paper.

## Advanced Options of DSDPFlex

There are advanced options in DSDPflex that allow manual adjustment or further development of the program.

- `--no_norm` let the program not normalize the intra-protein score (i.e. using the original Vina score, see more in the [paper](https://doi.org/10.26434/chemrxiv-2023-bcw0g-v2))
- `--norm_param <float>` modify the normalization parameter $c$ 
    the re-weighting factor of the intra-protein score ($\gamma$) will be  
    $$\gamma = c\times \min(f_\text{ligand} / f_\text{flex}, 1)$$  
    default $c = 1/2$
- `--rand_init_flex` randomly initialize flexible side-chain conformations. By default, DSDPflex keeps the initial side-chain conformations.
- `--rank_ligand_only` consider only ligand-related scores when ranking (&output) the poses.

## Cite this work

**DSDPFlex: Flexible-Receptor Docking with GPU Acceleration.**   
Chengwei Dong, Yu-Peng Huang, Xiaohan Lin, Hong Zhang, and Yi Qin Gao  
*Journal of Chemical Information and Modeling* **2024** *64* (22), 8537-8548  
DOI: [10.1021/acs.jcim.4c01715](https://doi.org/10.1021/acs.jcim.4c01715)


## LICENSE
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE (Version 3), you may not use this file except in compliance with the License. Any commercial use is not permitted.

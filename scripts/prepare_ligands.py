import os
from tqdm import tqdm

apobind = os.listdir("apobind_prepared")
for pdb in tqdm(apobind):
    last_dir = os.getcwd()
    os.chdir("apobind_prepared/" + pdb)
    os.system(
        f"/home/ayahc/Flex/MGLTools/MGLTools-1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l {pdb}_ligand.mol2 -o ligand.pdbqt"
    )
    os.chdir(last_dir)

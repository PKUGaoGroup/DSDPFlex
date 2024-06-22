import os
from tqdm import tqdm
from prepare_box import *

box_type = "DSDP"
apobind = os.listdir("apobind_prepared")
with open("apobind_cleaned.txt", "w") as f:
    f.writelines([i + "\n" for i in sorted(apobind)])
for pdb in apobind:
    last_dir = os.getcwd()
    os.chdir("apobind_prepared/" + pdb)

    boxdict = prepare_box_dict("ligand.pdbqt", "receptor_flex.pdbqt")
    if box_type == "DSDP":
        with open("box_dsdp.txt", "w") as f:
            boxline = [
                f"{i:.3f}"
                for i in [
                    *boxdict["ligbox_min"],
                    *boxdict["ligbox_max"],
                    *boxdict["box_min"],
                    *boxdict["box_max"],
                ]
            ]
            f.write(" ".join(boxline) + "\n")
    elif box_type == "vina":
        for method in ["flex", "rigid"]:
            with open(f"box_vina_{method}.txt", "w") as f:
                prefix = "ligbox" if method == "rigid" else "box"
                cx, cy, cz = boxdict[f"{prefix}_center"]
                sx, sy, sz = boxdict[f"{prefix}_size"]
                f.write(
                    f"center_x = {cx:.3f}\n"
                    f"center_y = {cy:.3f}\n"
                    f"center_z = {cz:.3f}\n"
                    f"size_x = {sx:.3f}\n"
                    f"size_y = {sy:.3f}\n"
                    f"size_z = {sz:.3f}\n"
                )
    os.chdir(last_dir)

def read_pdb_line(line: str):
    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    atom_no = int(line[6:12])
    atom_name = line[13:15]
    resn = line[17:20]
    chain = line[21]
    resi = line[22:26].strip(" ")
    atom_type = line[77]

    return {
        "atom_no": atom_no,
        "atom_name": atom_name,
        "atom_type": atom_type,
        "resn": resn,
        "chain": chain,
        "resi": resi,
        "x": x,
        "y": y,
        "z": z,
    }


def prepare_box_dict(ligand, flex, padding=4.0):
    import numpy as np

    with open(ligand) as l, open(flex) as f:
        lig_x, lig_y, lig_z = [], [], []
        for line in l.readlines():
            if "ATOM" in line or "HETATM" in line:
                atom = read_pdb_line(line)
                if atom["atom_type"] == "H":
                    continue
                lig_x.append(atom["x"])
                lig_y.append(atom["y"])
                lig_z.append(atom["z"])
        ligbox_min = np.array([np.min(lig_x), np.min(lig_y), np.min(lig_z)]) - padding
        ligbox_max = np.array([np.max(lig_x), np.max(lig_y), np.max(lig_z)]) + padding
        flex_x, flex_y, flex_z = [], [], []
        for line in f.readlines():
            if "ATOM" in line or "HETATM" in line:
                atom = read_pdb_line(line)
                if atom["atom_type"] == "H":
                    continue
                flex_x.append(atom["x"])
                flex_y.append(atom["y"])
                flex_z.append(atom["z"])
        flexbox_min = (
            np.array([np.min(flex_x), np.min(flex_y), np.min(flex_z)]) - padding
        )
        flexbox_max = (
            np.array([np.max(flex_x), np.max(flex_y), np.max(flex_z)]) + padding
        )
    # large box
    box_min = np.array([min(ligbox_min[i], flexbox_min[i]) for i in range(3)])
    box_max = np.array([max(ligbox_max[i], flexbox_max[i]) for i in range(3)])
    # vina style output
    ligbox_center = (ligbox_min + ligbox_max) / 2
    ligbox_size = ligbox_max - ligbox_min
    box_center = (box_min + box_max) / 2
    box_size = box_max - box_min
    return {
        "ligbox_min": ligbox_min.tolist(),
        "ligbox_max": ligbox_max.tolist(),
        "ligbox_center": ligbox_center.tolist(),
        "ligbox_size": ligbox_size.tolist(),
        "box_min": box_min.tolist(),
        "box_max": box_max.tolist(),
        "box_center": box_center.tolist(),
        "box_size": box_size.tolist(),
    }

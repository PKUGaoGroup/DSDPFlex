class flexres:

    def __init__(self, receptor, ligand):
        self.res_list = []
        self.res_list_reduced = []  # no ALA, GLY
        self.res_adt_style = []
        self.res_num = 0
        self.receptor = receptor
        self.ligand = ligand

    def get(self, format="ADT", contact_dist=3.5, exclude_metal_dist=2.5):
        """find contact residues by:
        receptor side chains within `contact_dist` of ligand heavy atoms (!h.)
        and not within `exclude_metal_dist` of any metal atoms

        Args:
            format (str, optional): AutoDockTools arg. Defaults to "ADT".
            contact_dist (float, optional):  Defaults to 3.5.
            exclude_metal_dist (float, optional):  Defaults to 2.5.

        Returns:
            defined by `format`
        """
        self.res_list = self.find_contact(contact_dist, exclude_metal_dist)
        self.res_list_reduced = self.reduce(self.res_list)
        self.res_num = len(self.res_list_reduced)
        self.res_adt_style = self.flex_adt_style(self.res_list_reduced)
        if format == "ADT":
            return self.res_adt_style

    def reduce(self, flex):
        flex_out = []
        for r in flex:
            if "ALA" in r or "GLY" in r:
                continue
            flex_out.append(r)
        flex_out = list(set(flex_out))
        return flex_out

    def flex_adt_style(self, flex: list):
        """
        Use commas to separate 'full names' which uniquely identify residues:
        hsg1:A:ARG8_ILE84,hsg1:B:THR4
        """

        chain_res = {}
        for r in flex:
            ch, resn, resi = r
            if ch not in chain_res.keys():
                chain_res[ch] = []
            chain_res[ch].append(resn + resi)

        out = []
        for ch in chain_res.keys():
            if ch == "":
                # no chain name
                out.append("_".join(chain_res[ch]))
                break
            else:
                outstr = "_".join(chain_res[ch])
                out.append("receptor:" + ch + ":" + outstr)
        return ",".join(out)

    def find_contact(self, contact_dist=3.5, exclude_metal_dist=2.5) -> list:
        """find contact residues by:
        receptor side chains within `contact_dist` of ligand heavy atoms (!h.)
        and not within `exclude_metal_dist` of any metal atoms

        Args:
            contact_dist (float, optional):  Defaults to 3.5.
            exclude_metal_dist (float, optional):  Defaults to 2.5.

        Returns:
            list: a list of residue id tuple (chain, resn, resi)
        """
        from pymol import cmd
        import os

        assert os.path.exists(self.receptor) and os.path.exists(self.ligand)
        cmd.delete("all")
        cmd.load(self.receptor, "rec")
        cmd.load(self.ligand, "lig")
        # only heavy
        cmd.remove("h.")
        cmd.select(
            "flexres",
            f"byres ((rec and polymer.protein and sc.) within {contact_dist} of (lig &! h.) )",
        )
        cmd.select(
            "metalres",
            f"byres ((rec and polymer.protein and sc.) within {exclude_metal_dist} of metals)",
        )
        near_ligand, near_metal = [], []
        cmd.iterate(
            "flexres and name CA",
            "near_ligand.append((chain, resn, resi))",
            space={"near_ligand": near_ligand},
        )
        cmd.iterate(
            "metalres and name CA",
            "near_metal.append((chain, resn, resi))",
            space={"near_metal": near_metal},
        )
        if len(near_ligand) != len(set(near_ligand)):
            print("found duplicate residues")

        return list(set(near_ligand) - set(near_metal))

    def call_adt_preparation(
        self,
        PATH="~/Flex/MGLTools/MGLTools-1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/",
        output_name="receptor.pdbqt",
        add_hydrogens=False,
    ):
        import os

        appendix = ""
        if add_hydrogens:
            appendix += " -A hydrogens"
        output_path, in_receptor_name = os.path.split(self.receptor)
        cwd = os.getcwd()
        os.chdir(output_path)  # change to workdir
        # make receptor.pdbqt
        os.system(
            f"{PATH}prepare_receptor4.py -r {in_receptor_name} -o {output_name} {appendix}; {PATH}prepare_flexreceptor4.py -r {output_name} -s {self.res_adt_style}"
        )

        # make receptor_rigid.pdbqt receptor_flex.pdbqt
        os.chdir(cwd)

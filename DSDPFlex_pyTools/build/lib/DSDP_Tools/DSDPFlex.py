import argparse
import subprocess
import os
import time


class DSDPFlex:
    def __init__(self):
        self.flex_mode = False
        self.rescore = ""
        self.config = dict()  # config of DSDP task
        self.dsdp_cmd = ""
        self.pose_record = []
        self.temp_top_n = 20
        self.final_top_n = 10
        self.verbose = False

    def build_config(self, args: dict):
        """
        Build DSDPFlex.config and command line
        Args:
            args (dict): var(args)
        """
        if args["flex"]:
            self.flex_mode = True
        if args["rescore"] == "gnina":
            self.rescore = "gnina"
            self.final_top_n = args["top_n"]
            args["top_n"] = self.temp_top_n
        if args["verbose"]:
            self.verbose = True
        config_cmdlike = ["DSDPflex"]

        # add to DSDP command (no --rescore)
        for k in args.keys():
            if k in {"rescore", "verbose"}:
                continue
            if args[k]:  # not None, True
                self.config[k] = args[k]
                config_cmdlike.append(f"--{k}")
                # list or str
                if isinstance(args[k], list):
                    # config_cmdlike.append(" ".join([str(i) for i in args[k]]))
                    config_cmdlike += [str(i) for i in args[k]]
                elif isinstance(args[k], bool):
                    pass
                else:
                    config_cmdlike.append(str(args[k]))
        # self.dsdp_cmd = " ".join(config_cmdlike)
        self.dsdp_cmd = config_cmdlike  # for shell=False run

    def _call_DSDP(self) -> bool:
        """
        Call DSDP for docking:
        DSDP should be in PATH
        """
        dsdp_result = subprocess.run(
            self.dsdp_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        print(dsdp_result.stdout)
        if dsdp_result.stderr:
            print(dsdp_result.stderr)
            return False
        else:
            return True

    def _call_GNINA_rescore(self):
        """
        Call GNINA for rescoring:
        GNINA should be in PATH
        > recorded in self.pose_record
        """
        receptor = self.config["protein"]
        if self.verbose:
            print("__#_|__CNNscore__|__CNNaffinity__")
        for i in range(self.temp_top_n):  # if can parallelize?
            ligand, flex = (
                self.pose_record[i]["ligand_p"],
                self.pose_record[i]["flex_p"],
            )
            gnina_result = subprocess.run(
                ["gnina", "-r", receptor, "-l", ligand, "--flex", flex, "--score_only"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
            # too much out
            # print(gnina_result.stdout)
            if gnina_result.stderr:
                print(gnina_result.stderr)
            lines = [line.strip() for line in gnina_result.stdout.split("\n")]
            CNNscore = [line.split()[-1] for line in lines if "CNNscore" in line]
            CNNaffinity = [line.split()[-1] for line in lines if "CNNaffinity" in line]
            self.pose_record[i]["CNNscore"] = float(CNNscore[0])
            self.pose_record[i]["CNNaffinity"] = float(CNNaffinity[0])
            if self.verbose:
                print(" %2d |  %s   |    %s " % (i, CNNscore[0], CNNaffinity[0]))

    def _split_poses(self, infile, outdir, prefix):
        """split_poses"""
        lines_store = {}  # store pdbqt lines
        curr_model = 0
        suffix = os.path.splitext(infile)[-1]  # use same suffix
        with open(infile) as f:
            for l in f.readlines():
                if "MODEL" in l:
                    num_model = l.split()[1]
                    curr_model = num_model
                    lines_store[curr_model] = []
                elif "ENDMDL" in l:
                    continue
                else:
                    lines_store[curr_model].append(l)
        splitted_list = []
        for k in lines_store.keys():
            sp_file = os.path.join(outdir, f"{prefix}_{k}{suffix}")
            with open(sp_file, "w") as f:
                f.writelines(lines_store[k])
            splitted_list.append(sp_file)
        return splitted_list

    def _make_temp_poses(self):
        """split DSDP-output poses into a temp dir"""
        out_dir, out_file = os.path.split(self.config["out"])
        temp_dir = os.path.join(out_dir, "temp_poses")
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        lig_split = self._split_poses(self.config["out"], temp_dir, "ligand")
        flex_split = self._split_poses(self.config["out_flex"], temp_dir, "flex")
        if len(lig_split) != self.temp_top_n or len(flex_split) != self.temp_top_n:
            if len(lig_split) != len(flex_split):
                print(
                    "Warning: #model of ligand and flex inconsistent"
                )  # do not simply warn
            else:
                self.temp_top_n = len(lig_split)  # sometimes it ouput less

        for i in range(self.temp_top_n):
            self.pose_record.append({"ligand_p": lig_split[i], "flex_p": flex_split[i]})

    def _rearrange_poses(self):
        """
        rearrange poses based on GNINA's CNNscore
        """
        pose_sorted = sorted(
            self.pose_record, key=lambda x: x["CNNscore"], reverse=True
        )  # sort rescored poses
        lig_name_split = os.path.splitext(self.config["out"])
        flex_name_split = os.path.splitext(self.config["out_flex"])
        lig_out_rescore = lig_name_split[0] + "_rescored" + lig_name_split[1]
        flex_out_rescore = flex_name_split[0] + "_rescored" + flex_name_split[1]

        with open(lig_out_rescore, "w") as f1, open(flex_out_rescore, "w") as f2:
            for i in range(self.final_top_n):
                lines_load = [
                    f"MODEL {i+1}\n",
                    f"REMARK CNNscore    {pose_sorted[i]['CNNscore']:.6f}\n",
                    f"REMARK CNNaffinity {pose_sorted[i]['CNNaffinity']:.6f}\n",
                ]
                with open(pose_sorted[i]["ligand_p"]) as f:
                    f1.writelines(lines_load + f.readlines() + ["ENDMDL\n"])
                with open(pose_sorted[i]["flex_p"]) as f:
                    f2.writelines(lines_load + f.readlines() + ["ENDMDL\n"])

    def run_dock(self):
        print("call DSDPFlex for docking ...")
        DSDPstatus = self._call_DSDP()
        if DSDPstatus and self.rescore == "gnina":
            # TODO handle error here?
            self._make_temp_poses()
            print("call GNINA for rescoring ...")
            self._call_GNINA_rescore()
            print("writing rearranged poses ...")
            self._rearrange_poses()


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(
        description="Flexible Docking",
    )
    # input
    parser.add_argument(
        "--ligand", required=True, help="ligand input PDBQT file [REQUIRED]"
    )
    parser.add_argument(
        "--protein", required=True, help="protein rigid input PDBQT file [REQUIRED]"
    )
    parser.add_argument("--flex", help="protein flex input PDBQT file")
    # search space
    parser.add_argument(
        "--box_min",
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="box_min x y z (Angstrom) [REQUIRED]",
        type=float,
    )
    parser.add_argument(
        "--box_max",
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="box_min x y z (Angstrom) [REQUIRED]",
        type=float,
    )
    parser.add_argument(
        "--ligbox_min",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="ligbox_min x y z (Angstrom)",
        type=float,
    )
    parser.add_argument(
        "--ligbox_max",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="ligbox_min x y z (Angstrom)",
        type=float,
    )
    # output
    parser.add_argument("--out", help="ligand poses output")
    parser.add_argument("--out_flex", help="flexible side chain poses output")
    parser.add_argument("--log", help="output log")
    # search settings
    parser.add_argument(
        "--exhaustiveness",
        default=384,
        type=int,
        help="number of GPU threads (number of copies)",
    )
    parser.add_argument(
        "--search_depth",
        default=40,
        type=int,
        help="number of searching steps for every copy",
    )
    parser.add_argument(
        "--top_n", default=10, type=int, help="number of desired output poses"
    )
    parser.add_argument("--use_rotamer", action="store_true")
    parser.add_argument(
        "--kernel_type",
        type=int,
        help="1: total energy kernel; 0: decoupled energy kernel",
        choices=(0, 1),
    )
    parser.add_argument(
        "--rescore",
        choices=["gnina"],
        help="call a rescore method; the rescored output will have a suffix _rescored.pdbqt",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    dsdpflex = DSDPFlex()
    print(vars(args))
    dsdpflex.build_config(vars(args))
    # Docking procedure
    dsdpflex.run_dock()
    print("DSDPFlex: total time %.3f s" % (time.time() - t0))


if __name__ == "__main__":
    main()

from spyrmsd import io, rmsd
import numpy as np
import os
import argparse
import time
import pandas as pd


def append_filename(inpath, something: str):
    sp = os.path.splitext(inpath)
    new_taskpath = sp[0] + something + sp[1]
    return new_taskpath


def convert_vina_lig(inpath):
    outpath = append_filename(inpath, "_lig")
    ignore_state = False
    with open(inpath) as f, open(outpath, "w") as g:
        for l in f.readlines():
            if "BEGIN_RES" in l:
                ignore_state = True
                continue
            if "END_RES" in l:
                ignore_state = False
                continue
            if not ignore_state:
                g.write(l)
    return outpath


def readout_task_rmsd(taskpath: str, refdir=None, symm=True) -> list:
    """readout_RMSD for 1 task

    Args:
        taskpath (str): path to output
        refdir (str, optional): _description_. Defaults to None.
        symm (bool, optional): symmetric considering. Defaults to True.

    Returns:
        list: RMSD
    """
    if refdir == None:
        refdir = taskpath
    task_RMSDs = []

    try:
        if "vina" in taskpath.lower():
            taskpath = convert_vina_lig(taskpath)

        ref = io.loadmol(refdir)
        mols = io.loadallmols(taskpath)
        # remove H
        for mol in mols:
            mol.strip()
        ref.strip()
        coords_ref = ref.coordinates
        anum_ref = ref.atomicnums
        adj_ref = ref.adjacency_matrix
        coords = [mol.coordinates for mol in mols]
        anum = mols[0].atomicnums
        adj = mols[0].adjacency_matrix
        if symm:
            # run a symmetric corrected RMSD
            task_RMSDs = rmsd.symmrmsd(
                coords_ref,
                coords,
                anum_ref,
                anum,
                adj_ref,
                adj,
            )
        else:
            task_RMSDs = []
            for i in range(len(coords)):
                task_RMSDs.append(
                    rmsd.rmsd(
                        coords_ref,
                        coords[i],
                        anum_ref,
                        anum,
                    )
                )
        return task_RMSDs
    # if output not exist / invalid
    except Exception as e:
        print("readout_task_rmsd()", e)
        if symm:
            print("trying simple RMSD")
            tryres = readout_task_rmsd(taskpath, refdir, symm=False)
            return tryres
        return [np.inf]


def main():
    # this can be reused
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docked_pdbqt",
        required=True,
        help=r"output{i}.pdbqt, everything before {i}.pdbqt",
    )
    parser.add_argument("--repeat", required=True, type=int, help="repeat times > 3")
    parser.add_argument(
        "--result_dir", required=True, help="$result_dir/taskid/$docked_pdbqt..."
    )
    parser.add_argument("--ref", default="ligand", help="ligand pose reference")
    parser.add_argument(
        "--rmsd_gate", type=float, default=2.5, help="rmsd < rmsd_gate is successful"
    )
    parser.add_argument(
        "--top_n", type=int, default=1, help="considering the best pose within top N"
    )
    parser.add_argument("--log")
    parser.add_argument("--record_csv")
    parser.add_argument("--long_csv")
    args = parser.parse_args()
    repeat = args.repeat
    topn = args.top_n
    output_pdbqt = args.docked_pdbqt
    result_dir = args.result_dir
    ref_pdbqt = args.ref
    task_list = os.listdir(result_dir)
    task_list = sorted(task_list)
    # topn_rmsd[topi] = [rmsd1, rmsd2, rmsd3, ...]
    topn_rmsd = [[] for _ in range(repeat)]
    taskid_list = []
    topn_list = []
    repeati_list = []
    rmsd_topn_ith_list = []
    for task in task_list:

        lastpath = os.getcwd()
        os.chdir(os.path.join(result_dir, task))
        for i in range(repeat):
            task_rmsd = readout_task_rmsd(
                output_pdbqt.replace("#", str(i + 1)) + ".pdbqt",
                ref_pdbqt + ".pdbqt",
            )[:topn]
            topn_rmsd[i].append(np.min(task_rmsd))
            # for long df
            for n in range(1, topn + 1):
                topn_list.append(n)
                taskid_list.append(task)
                repeati_list.append(i)
                rmsd_topn_ith_list.append(np.min(task_rmsd[:n]))
        os.chdir(lastpath)
    success_rates = []
    # write some log
    log = []
    log.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "\n")
    log.append(
        f"{args.docked_pdbqt} @ {args.result_dir}\n"
        f"RMSD < {args.rmsd_gate:.1f} angstrom\n"
    )
    # SR of each run
    for i in range(repeat):
        success = [j < args.rmsd_gate for j in topn_rmsd[i]]
        log.append(
            f"{success.count(True)} / {len(task_list)} = {success.count(True) / len(task_list):.3f}\n"
        )
        success_rates.append(success.count(True) / len(task_list))
    # print an average+-std
    log.append(
        f"SR = {np.mean(success_rates)*100:.2f} +- {np.std(success_rates, ddof=1)*100:.2f} %"
    )
    print(*log)
    # record something
    if args.log:
        with open(args.log, "w") as lg:
            lg.writelines(log)
    if args.record_csv:

        rmsd_dict = dict([(i, topn_rmsd[i]) for i in range(len(topn_rmsd))])
        df = pd.DataFrame(rmsd_dict, index=task_list)
        df.to_csv(args.record_csv)

    if args.long_csv:
        long_df = pd.DataFrame(
            {
                "system": taskid_list,
                "repeat": repeati_list,
                "topn": topn_list,
                "rmsd": rmsd_topn_ith_list,
            }
        )
        long_df.to_csv(args.long_csv)


if __name__ == "__main__":
    main()

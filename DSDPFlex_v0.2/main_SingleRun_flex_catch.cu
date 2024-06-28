#include "common.cuh"
#include "Kernel.cuh"
#include "DSDP_Task.cuh"
#include "Partial_Rigid_Small_Molecule.cuh"
#include "Flexible_Side_Chain.cuh"
#include "Neighbor_Grid.cuh"
#include "Rigid_Protein.cuh"
#include "Vina_Grid_Force_Field.cuh"
#include "Copy_pdbqt_Format.cuh"
#include "DSDP_Sort.cuh"
#include "Rotamer_Sampling.cuh"
#include "Call_Rescore.cuh"
#include <time.h>
#include <unistd.h>

#define OMP_TIME
#ifdef OMP_TIME
#include <omp.h>
#endif // OMP_TIME
#define NEW_PARSE
#ifdef NEW_PARSE
#include "CLI11.hpp"
#endif
#define DEBUG
#define TEST_FLEX
#define AND_SAMPLE
#define RAND_INIT_FLEX
#define BOOTSTRAP_KERNEL 0
#define TOTAL_KERNEL 1
std::vector<DSDP_TASK> task;
std::vector<PARTIAL_RIGID_SMALL_MOLECULE> ligand;
std::vector<FLEXIBLE_SIDE_CHAIN> flexsc;
RIGID_PROTEIN protein;
NEIGHBOR_GRID nl_grid;
VINA_GRID_FORCE_FIELD vgff;
COPY_pdbqt_FORMAT copy_pdbqt_flex;

int main(int argn, char *argv[])
{
	// for most situations, below parameters are suitable, no need for others to change
	// neighbor list related
	const VECTOR neighbor_grid_box_length = {400.f, 400.f, 400.f}; // 包住整个模拟原子的空间大小（和近邻表处理相关）
	const float cutoff = 8.f;									   // 截断
	const float neighbor_grid_skin = 2.f;

	// interpolation list number in one dimension
	const unsigned int protein_mesh_grid_one_dimension_numbers = 100;

	// a factor to restrain small ligand in searching space
	const float border_strenth = 100.f;

	// some vina force field parameters
	const float omega = 0.292300f * 0.2f; // C_inter/(1+omega*N_rot);
	const float beta = 0.838718f;		  // 600 K, 1/(kb*T)

	// the allowed longest running time in one searching turn
	const float max_allowed_running_time = 60.f; // in sec

	// below parameters can be changed while command line input, but default may be good
	unsigned int stream_numbers = 384; // 该版本每个stream就对应一个副本操作
	unsigned int search_depth = 40;	   // 每个副本尝试搜索的次数

	float box_length = 30.f;			  // another space restrain, manily because of the interpolation space limit, larger will slower(if keep interpolation precision in the same time)
	int max_record_numbers = 2000;		  // only consider top 2000 poses by energy sorting
	float rmsd_similarity_cutoff = 1.f;	  // a parameter to distinguish two different poses
	int desired_saving_pose_numbers = 50; // try to find the best 50 results to save

	int optim_iterations = 10; // 在每个循环内多次采样小分子或侧链

	float rotational_amp = 3.141592654f / 4;
	double time_begin = omp_get_wtime();
	// file name
	const int MAXLINE = 512;
	char ligand_name[MAXLINE];	// required
	char protein_name[MAXLINE]; // required
	float FLEX_GATE = 0.5;		// control Flex samplingx

	char protein_flex_name[MAXLINE]; // optional
	bool flex_mode = false;
	int ligbox_input = 0;
	VECTOR box_min, box_max;	   // required
	VECTOR ligbox_min, ligbox_max; // optional
	char out_pdbqt_name[MAXLINE] = "DSDP_out_ligand.pdbqt";
	char out_list_name[MAXLINE] = "DSDP_out.log";
	char fsc_out_pdbqt_name[MAXLINE] = "DSDP_out_flex.pdbqt";
	// single mode or batch mode?
	bool ligand_batch_mode = false;

	bool verbose = false;
	bool rotamer_mode = false;
	bool dynamic_depth = false;
	bool no_normalization = false;
	bool rescore = false;
	bool rank_only_ligand = false;
	bool random_init_flex = false;
	long acc_cnt = 0, rej_cnt = 0; // for debug
	int rotamer_sampl_steps = 0;
	const char *rotamer_lib_path = "../Try_Rotamer_lib/bbind02.May.lib";
	const char *rotamer_def_path = "../Try_Rotamer_lib/rotamer.def";
	int kernel_type = TOTAL_KERNEL;
	float FLEX_RATIO_PAR = 1.0; // 1.0 or 0.5

	bool randomize_only = false;
	bool debug = false;

	/* a new argparsing style */
	CLI::App app{"DSDP"};
	std::string ligand_string = "", protein_string = "", flex_string = "";
	std::string batch_list = "";
	std::string ligand_path_str = ""; // compatibale with DSDP
	std::string ligand_name_list = "";
	std::vector<TaskIO> task_io_list; // maintain this
	std::string out_string, out_flex_string, log_string;
	std::vector<float> vbox_min, vbox_max, vligbox_min, vligbox_max;

	app.description("DSDP: Deep Site and Docking Pose\n"
					"  This is the flexible docking program (DSDPFlex), docking with known binding site.\n"
					"  See more details at https://github.com/PKUGaoGroup/DSDPFlex\n");

	app.add_option("--ligand", ligand_string, "ligand input PDBQT file")
		->option_text("<pdbqt>")
		->check(CLI::ExistingFile)
		->group("Input");

	app.add_option("--protein", protein_string, "protein rigid-part input PDBQT file [REQUIRED]")
		->option_text("<pdbqt>")
		->required()
		->check(CLI::ExistingFile)
		->group("Input");

	app.add_option("--flex", flex_string, "protein flex-part input PDBQT file")
		->option_text("<pdbqt>")
		->check(CLI::ExistingFile)
		->group("Input");
	app.add_option("--ligand_batch", batch_list, "lines: <ligand> <out> <out_flex>")
		->option_text("<txt>")
		->check(CLI::ExistingFile)
		->group("Input");
	// compatible with DSDP_drug_list
	app.add_option("--box_min", vbox_min, "grid_box min: x y z (Angstrom) [REQUIRED]")
		->option_text("x y z")
		->required()
		->expected(3)
		->group("Search space");
	app.add_option("--box_max", vbox_max, "grid_box max: x y z (Angstrom) [REQUIRED]")
		->option_text("x y z")
		->required()
		->expected(3)
		->group("Search space");
	app.add_option("--ligbox_min", vligbox_min, "ligand_box min: x y z (Angstrom)")
		->option_text("x y z")
		->expected(3)
		->group("Search space");

	app.add_option("--ligbox_max", vligbox_max, "ligand_box max: x y z (Angstrom)")
		->option_text("x y z")
		->expected(3)
		->group("Search space");

	app.add_option("--out", out_string, "ligand poses output [=DSDP_out.pdbqt]")
		->option_text("<pdbqt>")
		->default_val(std::string("DSDP_out.pdbqt"))
		->group("Output");
	app.add_option("--out_flex", out_flex_string, "flexible side chain poses output [=DSDP_out_flex.pdbqt]")
		->option_text("<pdbqt>")
		->default_val(std::string("DSDP_out_flex.pdbqt"))
		->group("Output");

	app.add_option("--log", log_string, "log output [=DSDP_out.log]")
		->option_text("<log>")
		->default_val(std::string("DSDP_out.log"))
		->group("Output");
	app.add_option("--exhaustiveness", stream_numbers, "number of GPU threads (number of copies) [=384]")
		->default_val(384)
		->option_text("N")
		->group("Search settings");

	app.add_option("--search_depth", search_depth, "number of searching steps for every copy [=40]")
		->default_val(40)
		->option_text("N")
		->group("Search settings");
	app.add_option("--top_n", desired_saving_pose_numbers, "number of desired output poses [=10]")
		->default_val(10)
		->option_text("N")
		->group("Search settings");
	app.add_flag("--use_rotamer", rotamer_mode)->group("Search settings");
	app.add_flag("--dynamic_depth", dynamic_depth)->group("Search settings");
	app.add_option("--kernel_type")
		->default_val(TOTAL_KERNEL)
		->group("Search settings");
	app.add_flag("--randomize_only", randomize_only)->group("Misc");
	app.add_flag("--no_norm", no_normalization)->group("Misc");
	app.add_flag("--rand_init_flex", random_init_flex)->group("Misc");
	app.add_flag("--rescore", rescore)
		->group("Misc");
	app.add_flag("--rank_ligand_only", rank_only_ligand)->group("Misc");
	app.add_option("--norm_param", FLEX_RATIO_PAR, "flex_ratio = norm_param * min(g_ligand/f_flex, 1)")->default_val(0.5)->group("Misc");
	app.add_flag("--debug", debug)->group("Misc");

	// app.add_flag("--verbose", verbose);
	// app.add_flag("--rotamer", rotamer_mode);

	CLI11_PARSE(app, argn, argv);

	time_t now_time;
	time(&now_time);
	printf(
		"|\\(`|\\|)  DSDPFlex\n"
		"|/_)|/|   ver 0.2a\n");
	printf("START  > %s", ctime(&now_time));
	if (flex_string != "")
		flex_mode = true;

	if (ligand_string != "")
	{
		ligand_batch_mode = false;
		// only one task
		task_io_list = {TaskIO{ligand_string, out_string, out_flex_string}};
	}
	else if (batch_list != "")
	{
		// some tasks
		int task_num = Read_list_from_file(batch_list, &task_io_list);
		printf("DSDPFlex read %d tasks from %s\n", task_num, batch_list.c_str());
	}
	else
	{
		// fault
		printf("--ligand or --batch is neccessary!\n(exit)\n");
		return 1;
	}
	// not depend on task
	sscanf(protein_string.c_str(), "%s", protein_name);
	sscanf(flex_string.c_str(), "%s", protein_flex_name);
	sscanf(log_string.c_str(), "%s", out_list_name);

	box_min = {vbox_min[0], vbox_min[1], vbox_min[2]};
	box_max = {vbox_max[0], vbox_max[1], vbox_max[2]};

	if (vligbox_min.size() == 3 && vligbox_max.size() == 3)
	{
		ligbox_min = {vligbox_min[0], vligbox_min[1], vligbox_min[2]};
		ligbox_max = {vligbox_max[0], vligbox_max[1], vligbox_max[2]};

		if (ligbox_min.x < box_min.x || ligbox_min.y < box_min.y || ligbox_min.z < box_min.z ||
			ligbox_max.x > box_max.x || ligbox_max.y > box_max.y || ligbox_max.z > box_max.z)
		{
			printf("WARNING: ligbox should be smaller than box. ligbox is modified.\n");
			ligbox_min.x = fmax(box_min.x, ligbox_min.x);
			ligbox_min.y = fmax(box_min.y, ligbox_min.y);
			ligbox_min.z = fmax(box_min.z, ligbox_min.z);
			ligbox_max.x = fmin(box_max.x, ligbox_max.x);
			ligbox_max.y = fmin(box_max.y, ligbox_max.y);
			ligbox_max.z = fmin(box_max.z, ligbox_max.z);
		}
	}
	else
	{
		ligbox_min = box_min;
		ligbox_max = box_max;
		printf("using ligbox = box\n");
	}

	printf("* protein: %s\n* flex:    %s\n", protein_name, protein_flex_name);
	Rotamer *rotamer_sampler = new Rotamer();

	/* Initialization phase start */

	srand((int)time(0));
	cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleAuto);

	task.resize(stream_numbers); // GPU tasks
	for (int i = 0; i < stream_numbers; i = i + 1)
	{
		task[i].Initial();
	}

	/* rigid protein initialize */
	nl_grid.Initial(neighbor_grid_box_length, cutoff, neighbor_grid_skin);
	vgff.Initial(protein_mesh_grid_one_dimension_numbers, cutoff);
	protein.Initial_Protein_From_PDBQT(protein_name, neighbor_grid_box_length);
	nl_grid.gpu.Put_Atom_Into_Grid_Bucket(protein.atom_numbers, &protein.crd[0]);
	/*  flex chains initialize, if any */
	flexsc.resize(stream_numbers);
	flexsc[0].Initial_From_PDBQT(protein_flex_name);
	for (int i = 1; i < stream_numbers; i = i + 1)
	{
		flexsc[i].Copy_From_FLEXIBLE_SIDE_CHAIN(&flexsc[0]);
	}
	copy_pdbqt_flex.Initial(protein_flex_name);

	// all crds pushed by protein.move_vecs
	box_min = v_plus(box_min, protein.move_vec);
	box_max = v_plus(box_max, protein.move_vec);
	ligbox_min = v_plus(ligbox_min, protein.move_vec);
	ligbox_max = v_plus(ligbox_max, protein.move_vec);

	//  initialize box_length 10.07
	box_length = fmax(box_length, box_max.x - box_min.x);
	box_length = fmax(box_length, box_max.y - box_min.y);
	box_length = fmax(box_length, box_max.z - box_min.z);
	if (box_length > 30)
	{
		printf("WARNING: grid box is large\n");
	}

	/* protein grid box initialize */
	vgff.grid.Calculate_Protein_Potential_Grid(
		box_min, box_length,
		protein.atom_numbers, protein.d_vina_atom,
		nl_grid.grid_length_inverse, nl_grid.grid_dimension, nl_grid.gpu.neighbor_grid_bucket);

	/*Initialize Rotamer Library*/
	if (rotamer_mode)
	{
		try
		{
			// read residues list to rotamer_sampler
			// even if don't use rotamer sampling, it can give useful infomation
			rotamer_sampler->residue_list = flexsc[0].dof_residues;
			rotamer_sampler->residue_index = flexsc[0].dof_resid;
			rotamer_sampler->Initial_from_residue_list();

			rotamer_sampler->Initialize_lib(rotamer_lib_path); // read rotamer library
			rotamer_sampler->Initialize_def(rotamer_def_path);
			rotamer_sampler->Initial_dihedral_from_PDBQT(protein_name, protein_flex_name);
			printf("Rotamer lib initialized.\n");
		}
		catch (const std::exception &e)
		{
			std::cerr << e.what() << '\n';
			std::cerr << "Rotamer failed to initialize, use default sampling instead\n";
			rotamer_mode = false;
		}
	}
	// end of rotamer initialization

	if (debug)
	{
		printf("initialization check:\n");
		print_vec(box_min, "* box_min");
		print_vec(box_max, "* box_max");
		print_vec(ligbox_min, "* ligbox_min");
		print_vec(ligbox_max, "* ligbox_max");
		print_vec(protein.move_vec, "* protein.move_vec");
	}
	printf("protein initialization takes %lf s\n", omp_get_wtime() - time_begin);

	FILE *out_list = fopen_safely(out_list_name, "w"); // output log file
	// Docking Circulation
	for (int task_i = 0; task_i < task_io_list.size(); task_i++)
	{
		printf("Task #%d start >\n", task_i);
		double task_time_begin = omp_get_wtime();
		// set ligand/flex input/output files
		sscanf(task_io_list[task_i].lig_in.c_str(), "%s", ligand_name);
		sscanf(task_io_list[task_i].lig_out.c_str(), "%s", out_pdbqt_name);
		sscanf(task_io_list[task_i].fsc_out.c_str(), "%s", fsc_out_pdbqt_name);
		printf("* ligand:  %s\n", ligand_name);

		// try-catch here
		try
		{

			int dof_lig = 0;		   // degree of freedom of ligand
			int dof_flex = 0;		   // degree of freedom of flex chains
			int true_flex_freedom = 0; // true degree of freedom of flex chains
			float flex_ratio = 1;
			// it's safe to define here
			COPY_pdbqt_FORMAT copy_pdbqt_ligand;
			DSDP_SORT DSDP_sort;

			std::vector<VECTOR> lig_crd_record;						   // ligand coordinates
			std::vector<VECTOR> fsc_crd_record;						   // flex chains coordinates
			std::vector<INT_FLOAT> total_energy_record;				   // total energy record <- target
			std::vector<INT_FLOAT> ligand_energy_record;			   // energy record for ligand
			std::vector<INT_FLOAT> intra_protein_energy_record;		   // for energy_shift
			std::vector<int> search_numbers_record(stream_numbers, 0); // search numbers for every copy

			/* ligand initialize */
			ligand.clear();
			ligand.resize(stream_numbers);
			ligand[0].Initial_From_PDBQT(ligand_name);
			for (int i = 1; i < stream_numbers; i = i + 1)
			{
				ligand[i].Copy_From_PARTIAL_RIGID_SMALL_MOLECULE(&ligand[0]);
			}
			copy_pdbqt_ligand.Initial(ligand_name); // for output

			dof_lig = ligand[0].vina_gpu.u_freedom;
			// u_freedom - 6 is the true dof (without trans. rot.)
			dof_flex = flexsc[0].vina_gpu.u_freedom;
			true_flex_freedom = dof_flex - 6;
			// controlling the Intra-protein energy contibution
			FLEX_GATE = float(dof_flex) / float(dof_lig);
			flex_ratio = float(dof_lig) / float(true_flex_freedom);
			flex_ratio = FLEX_RATIO_PAR * fmin(flex_ratio, 1); // TODO param tweak

			// dynamic depth
			if (dynamic_depth)
			{
				search_depth = min(20 + (dof_lig + true_flex_freedom) / 2, 60);
			}
			if (no_normalization)
				flex_ratio = 1.0;
			if (debug)
			{
				printf("DoF_ligand  %2d \n", dof_lig);
				printf("DoF_flex    %2d \n", true_flex_freedom);
				printf("flex_ratio %.3f = min( %.1f * %d/%d, %.1f)\n",
					   flex_ratio, FLEX_RATIO_PAR, dof_lig, true_flex_freedom, FLEX_RATIO_PAR);
			}

			/* CONFORMATION INITIALIZTION */
			for (int i = 0; i < stream_numbers; i = i + 1)
			{
				VECTOR rand_vec = {(float)0.5f * rand() / RAND_MAX + 0.25f, (float)0.5f * rand() / RAND_MAX + 0.25f, (float)0.5f * rand() / RAND_MAX + 0.25f};
				// torsions
				for (int j = 0; j < dof_lig - 6; j = j + 1)
				{
					ligand[i].vina_gpu.h_u_crd[j] = 2.f * 3.141592654f * rand() / RAND_MAX;
				}
				// rotaion
				VECTOR rand_angle = uniform_rand_Euler_angles();
				ligand[i].vina_gpu.h_u_crd[dof_lig - 3] = rand_angle.z;
				ligand[i].vina_gpu.h_u_crd[dof_lig - 2] = rand_angle.y;
				ligand[i].vina_gpu.h_u_crd[dof_lig - 1] = rand_angle.x;

				// translation (ligbox)
				ligand[i].vina_gpu.h_u_crd[dof_lig - 6] = (ligbox_min.x + rand_vec.x * (ligbox_max.x - ligbox_min.x));
				ligand[i].vina_gpu.h_u_crd[dof_lig - 5] = (ligbox_min.y + rand_vec.y * (ligbox_max.y - ligbox_min.y));
				ligand[i].vina_gpu.h_u_crd[dof_lig - 4] = (ligbox_min.z + rand_vec.z * (ligbox_max.z - ligbox_min.z));

				ligand[i].vina_gpu.last_accepted_energy = (float)1e3; // large energy
				task[i].Assign_Status(DSDP_TASK_STATUS::EMPTY);

				memcpy(ligand[i].vina_gpu.h_last_accepted_u_crd, ligand[i].vina_gpu.h_u_crd, sizeof(float) * dof_lig);

				// (re-)initialize flex (default)
				for (int j = 0; j < dof_flex - 6; j = j + 1)
				{
					flexsc[i].vina_gpu.h_u_crd[j] = 0;
				}
				// only if random_init_flex, do vina-style flex initialization
				if (random_init_flex)
				{
					for (int j = 0; j < dof_flex - 6; j = j + 1)
					{
						flexsc[i].vina_gpu.h_u_crd[j] = 2.f * 3.141592654f * rand() / RAND_MAX;
					}
				}
				else if (rotamer_mode)
				{
					for (int j = 0; j < dof_flex - 6; j = j + 1)
					{
						rotamer_sampler->Torsion_initialize(flexsc[i].vina_gpu.h_u_crd, dof_flex - 6);
					}
				}

				// recover translational crd.
				// plus protein's move_vec
				VECTOR flex_init = {flexsc[i].move_vec.x, flexsc[i].move_vec.y, flexsc[i].move_vec.z};
				// printf("FLEX Init0, %.3f, %.3f, %.3f\n", flex_init.x, flex_init.y, flex_init.z);
				flex_init.x += protein.move_vec.x;
				flex_init.y += protein.move_vec.y;
				flex_init.z += protein.move_vec.z;
				flexsc[i].vina_gpu.h_u_crd[dof_flex - 6] = flex_init.x;
				flexsc[i].vina_gpu.h_u_crd[dof_flex - 5] = flex_init.y;
				flexsc[i].vina_gpu.h_u_crd[dof_flex - 4] = flex_init.z;

				flexsc[i].vina_gpu.last_accepted_energy = (float)1e3;

				memcpy(flexsc[i].vina_gpu.h_last_accepted_u_crd, flexsc[i].vina_gpu.h_u_crd, sizeof(float) * (dof_flex));

			} // End of conformation initialization
			printf("initialization takes %lf s\n", omp_get_wtime() - task_time_begin);

			/* SEARCHING */
#ifdef OMP_TIME
			double time_start = omp_get_wtime();
#endif // OMP_TIME

			int optim_kernel_version[stream_numbers]; // 规定优化kernel: 0/1
			bool large_rotation_sampled[stream_numbers];

			printf("running first search ... ");
			lig_crd_record.clear();
			fsc_crd_record.clear();
			total_energy_record.clear();
			ligand_energy_record.clear();
			intra_protein_energy_record.clear();
			/* MAIN SEARCHING ROUTINE */
			cudaDeviceSynchronize();
			while (true)
			{
				bool is_ok_to_break = true;
				for (int i = 0; i < stream_numbers; i = i + 1)
				{
					if (task[i].Is_empty()) // 如果当前stream无任务
					{

						if (task[i].Get_Status() == DSDP_TASK_STATUS::MINIMIZE_STRUCTURE) // 如果当前stream是做完了一次最优化
						{

							/* METROPOLIS ACCEPT/REJECT */
							// 现在在dof_flex-1存放inter energy
							// total_energy = lig + intra-fsc
							float inter_lig_fsc_energy = flexsc[i].vina_gpu.h_u_crd[dof_flex - 1];
							float intra_protein_energy = flexsc[i].vina_gpu.h_u_crd[dof_flex];
							float ligand_energy = ligand[i].vina_gpu.h_u_crd[dof_lig];
							float total_current_energy = ligand_energy + intra_protein_energy;
							float probability = expf(fminf(beta *
															   (ligand[i].vina_gpu.last_accepted_energy +
																flexsc[i].vina_gpu.last_accepted_energy -
																total_current_energy),
														   0.f));

							if (probability > (float)rand() / RAND_MAX) // ACCEPT
							{
								// record ligand
								ligand[i].vina_gpu.last_accepted_energy = ligand_energy;
								for (int j = 0; j < ligand[i].atom_numbers; j = j + 1)
								{
									lig_crd_record.push_back(ligand[i].vina_gpu.h_vina_atom[j].crd);
								}
								// record flex side chain
								flexsc[i].vina_gpu.last_accepted_energy = intra_protein_energy;
								flexsc[i].vina_gpu.last_accepted_inter_energy = inter_lig_fsc_energy;
								for (int j = 0; j < flexsc[i].atom_numbers; j = j + 1)
								{
									fsc_crd_record.push_back(flexsc[i].vina_gpu.h_vina_atom[j].crd);
								}
								memcpy(ligand[i].vina_gpu.h_last_accepted_u_crd, ligand[i].vina_gpu.h_u_crd, sizeof(float) * dof_lig);
								memcpy(flexsc[i].vina_gpu.h_last_accepted_u_crd, flexsc[i].vina_gpu.h_u_crd, sizeof(float) * dof_flex);
								// record total energy
								total_energy_record.push_back({(int)total_energy_record.size(), total_current_energy});
								intra_protein_energy_record.push_back({(int)intra_protein_energy_record.size(), intra_protein_energy});
								ligand_energy_record.push_back({(int)ligand_energy_record.size(), ligand_energy});

								acc_cnt += 1;
							}
							else // REJECT
							{
								memcpy(ligand[i].vina_gpu.h_u_crd, ligand[i].vina_gpu.h_last_accepted_u_crd, sizeof(float) * dof_lig);
								memcpy(flexsc[i].vina_gpu.h_u_crd, flexsc[i].vina_gpu.h_last_accepted_u_crd, sizeof(float) * dof_flex);
								rej_cnt += 1;
							}
						}
						/* SAMPLING & BB OPTIMIZATION */

						// sampling ligand AND flex chains
						int rand_int = rand() % dof_lig;
						if (rand_int < dof_lig - 3)
						{
							if (rand_int < dof_lig - 6)
							{
								// torsion angles
								ligand[i].vina_gpu.h_u_crd[rand_int] = 2.f * 3.141592654f * ((float)rand() / RAND_MAX);
							}
							else
							{
								// translational vector
								ligand[i].vina_gpu.h_u_crd[rand_int] += 1.f * (2.f * ((float)rand() / RAND_MAX) - 1.f);
							}
						}
						else // rotation angle
						{
							// \theta ~ (0, 2A/r), where r is gyration_radius
							// VECTOR rand_angle = uniform_rand_Euler_angles_range(rotational_amp);
							VECTOR rand_angle = uniform_rand_Euler_angles(); // FIXME
							ligand[i].vina_gpu.h_u_crd[dof_lig - 3] = rand_angle.z;
							ligand[i].vina_gpu.h_u_crd[dof_lig - 2] = rand_angle.y;
							ligand[i].vina_gpu.h_u_crd[dof_lig - 1] = rand_angle.x;
						}

						// rotamer sampling
						if (rotamer_mode)
							rotamer_sampler->Torsion_sample(flexsc[i].vina_gpu.h_u_crd, dof_flex - 6);
						else
						{
							int rand_int_0 = rand() % (dof_flex - 6);
							flexsc[i].vina_gpu.h_u_crd[rand_int_0] = 2.f * 3.141592654f * ((float)rand() / RAND_MAX);
						}
						// after sampling
						// copy *(dof_lig+1) because u_crd[dof_lig] is current energy
						cudaMemcpyAsync(flexsc[i].vina_gpu.u_crd, flexsc[i].vina_gpu.h_u_crd, sizeof(float) * (dof_flex + 1), cudaMemcpyHostToDevice, task[i].Get_Stream());
						cudaMemcpyAsync(ligand[i].vina_gpu.u_crd, ligand[i].vina_gpu.h_u_crd, sizeof(float) * (dof_lig + 1), cudaMemcpyHostToDevice, task[i].Get_Stream());

						/* The BB2 optimization procedure
						 *1 step optimization, calculating ligand and flex chains at the same time
						 * providing a 'physical' updating strategy */

						Optimize_All_Structure_BB2_Direct_Pair_Device<<<1, 128, sizeof(float) * 48, task[i].Get_Stream()>>>(
							// protein grid
							vgff.grid.texObj_for_kernel, border_strenth, vgff.grid_length_inverse,
							box_min, box_max, ligbox_min, ligbox_max, cutoff,
							ligand[i].vina_gpu, flexsc[i].vina_gpu,
							&ligand[i].vina_gpu.u_crd[dof_lig],		 // ligand energy
							&flexsc[i].vina_gpu.u_crd[dof_flex],	 // intra-protein energy
							&flexsc[i].vina_gpu.u_crd[dof_flex - 1], // inter lig-fsc energy
							flex_ratio);

						/* This Kernel perform decoupled optimization */
						/* decrepated */
						search_numbers_record[i] += 1;
						// 1 step memcpy after optim
						cudaMemcpyAsync(ligand[i].vina_gpu.h_u_crd, ligand[i].vina_gpu.u_crd, sizeof(float) * (dof_lig + 1), cudaMemcpyDeviceToHost, task[i].Get_Stream());
						cudaMemcpyAsync(ligand[i].vina_gpu.h_vina_atom, ligand[i].vina_gpu.d_vina_atom, sizeof(VINA_ATOM) * ligand[i].atom_numbers, cudaMemcpyDeviceToHost, task[i].Get_Stream());

						cudaMemcpyAsync(flexsc[i].vina_gpu.h_u_crd, flexsc[i].vina_gpu.u_crd, sizeof(float) * (dof_flex + 1), cudaMemcpyDeviceToHost, task[i].Get_Stream());
						cudaMemcpyAsync(flexsc[i].vina_gpu.h_vina_atom, flexsc[i].vina_gpu.d_vina_atom, sizeof(VINA_ATOM) * flexsc[i].atom_numbers, cudaMemcpyDeviceToHost, task[i].Get_Stream());

						task[i].Assign_Status(DSDP_TASK_STATUS::MINIMIZE_STRUCTURE);
						task[i].Record_Event();
					} // end of task assignment

					if (search_numbers_record[i] < search_depth)
					{
						is_ok_to_break = false;
					}

				} // for every stream

#ifdef OMP_TIME
				if (omp_get_wtime() - time_start > max_allowed_running_time)
				{
					is_ok_to_break = true;
				}
#endif // OMP_TIME
				if (is_ok_to_break)
				{
					break;
				}
			} // while 第一次搜索
			cudaDeviceSynchronize();
			printf("done.\n");
#ifdef OMP_TIME
			time_start = omp_get_wtime() - time_start;
#endif // OMP_TIME

			if (debug)
			{
				double acc_rate = (double)acc_cnt / (double)(rej_cnt + acc_cnt);
				printf("accept rate is %.4f\n", acc_rate);
			}
			/* OUTPUT PHASE */
			if (total_energy_record.size() == 0)
			{
				perror("DSDP didn't find valid solutions. Check the settings.\n");
				break; // quit this task (ignore output)
			}
			// sort by what? = total_energy
			//            or = ligand_energy
			std::vector<INT_FLOAT> energy_record_ranking = total_energy_record;

			if (rank_only_ligand)
				energy_record_ranking = ligand_energy_record;

			sort(energy_record_ranking.begin(), energy_record_ranking.end(), cmp);

			// energy_shift = intramolecular energy
			float energy_shift = 0.f;
			float intra_ligand = 0.f;
			float intra_protein = 0.f;
			VECTOR *ligand_crd = &lig_crd_record[(size_t)energy_record_ranking[0].id * (ligand[0].atom_numbers)];
			// calc. intra-ligand energy again
			for (int i = 0; i < ligand[0].atom_numbers; i = i + 1)
			{
				VINA_ATOM atom_j;
				VINA_ATOM atom_i = ligand[0].vina_gpu.h_vina_atom[i];
				atom_i.crd = ligand_crd[i];
				int inner_list_start = i * ligand[0].atom_numbers;
				int inner_numbers = ligand[0].vina_gpu.h_inner_neighbor_list[inner_list_start];
				for (int k = 1; k <= inner_numbers; k = k + 1)
				{
					int j = ligand[0].vina_gpu.h_inner_neighbor_list[inner_list_start + k];
					atom_j = ligand[0].vina_gpu.h_vina_atom[j];
					atom_j.crd = ligand_crd[j];
					float2 temp = Vina_Pair_Interaction(atom_i, atom_j);
					energy_shift += temp.y;
				}
			}
			// energy_shift /= (1.f + omega * ligand[0].num_tor);
			intra_ligand = energy_shift;

			//  find recorded intra-protein energy (fsc-fsc + fsc-protein)
			for (int i = 0; i < intra_protein_energy_record.size(); i += 1)
			{
				// 匹配的能量
				if (intra_protein_energy_record[i].id == energy_record_ranking[0].id)
				{
					energy_shift += intra_protein_energy_record[i].energy; // already normalized
					intra_protein = intra_protein_energy_record[i].energy;
					// printf("intra protein energy is %f\n", intra_protein_energy_record[i].energy);
					break;
				}
			}
			if (debug)
			{
				printf("Top-1 affinity = %.3f", energy_record_ranking[0].energy);
				printf(" - %.3f = %.3f\n", energy_shift, energy_record_ranking[0].energy - energy_shift);
				printf("* ligand-protein %6.3f\n"
					   "* intra-ligand   %6.3f\n"
					   "* intra-protein  %6.3f\n",
					   energy_record_ranking[0].energy - energy_shift, intra_ligand, intra_protein);
			}
			// end of calc. energy shift

			VECTOR move_vec = {-protein.move_vec.x, -protein.move_vec.y, -protein.move_vec.z}; // to move all coordinates back
			if (rescore)
			{
				// new records
				std::vector<INT_FLOAT> rescored_energy_ranking;
				Rescore rescorer;
				rescorer.initialize_from_input(protein_string, out_pdbqt_name, out_flex_string);

				for (int i = 0; i < desired_saving_pose_numbers; i += 1)
				{
					// open temp files
					FILE *fp_ligand = rescorer.fp_temp_ligand_out();
					FILE *fp_flex = rescorer.fp_temp_flex_out();
					// write crds to files
					copy_pdbqt_ligand.Append_Frame_To_Opened_pdbqt(
						fp_ligand, &lig_crd_record[(size_t)i * (ligand[0].atom_numbers)], move_vec);
					copy_pdbqt_flex.Append_Frame_To_Opened_pdbqt(
						fp_flex, &fsc_crd_record[(size_t)i * (flexsc[0].atom_numbers)], move_vec);
					rescorer.close_files();
					// call gnina rescore
					float gnina_score = rescorer.call_gnina();
					rescored_energy_ranking.push_back({energy_record_ranking[i].id, gnina_score});
				}
				energy_record_ranking = rescored_energy_ranking;
				sort(energy_record_ranking.begin(), energy_record_ranking.end(), cmp);
			}

			// 现在以total_energy 排序的结果作为输出的索引
			DSDP_sort.Sort_Structures(
				ligand[0].atom_numbers, flexsc[0].atom_numbers,
				&ligand[0].atomic_number[0], // for rmsd
				std::min(max_record_numbers, (int)energy_record_ranking.size()),
				&lig_crd_record[0], &fsc_crd_record[0], &energy_record_ranking[0],
				rmsd_similarity_cutoff, desired_saving_pose_numbers, desired_saving_pose_numbers);

			// like Vina, calculate RMSD to top-1 pose
			printf("Top modes:\n"
				   "__#_|__affinity__|__RMSD-Top1___\n");
			for (int i = 0; i < min(DSDP_sort.selected_numbers, 5); i += 1)
			{
				float rmsd_to_best = calcualte_heavy_atom_rmsd(
					ligand[0].atom_numbers,
					&DSDP_sort.selected_lig_crd[(size_t)i * (ligand[0].atom_numbers)],
					&DSDP_sort.selected_lig_crd[0],
					&ligand[0].atomic_number[0]);
				printf("%3d |  %8.3f  |  %8.3f \n", i,
					   (DSDP_sort.selected_energy[i] - energy_shift) / (1.f + omega * ligand[0].num_tor),
					   rmsd_to_best);
			}
			printf("...\n");

			FILE *ligand_out_pdbqt = fopen_safely(out_pdbqt_name, "w");	 // file recording poses
			FILE *fsc_out_pdbqt = fopen_safely(fsc_out_pdbqt_name, "w"); // file recording poses
			printf("writing results ... ");
			// FIXME energy_record_ranking[i].energy - energy_shift or
			// DSDP_sort.selected_energy[i] - energy_shift ?
			for (int i = 0; i < DSDP_sort.selected_numbers; i += 1)
			{
				// printf("Append_Frame_To_Opened_pdbqt_standard()");
				float score_out_i = (DSDP_sort.selected_energy[i] - energy_shift) / (1.f + omega * ligand[0].num_tor);
				copy_pdbqt_flex.Append_Frame_To_Opened_pdbqt_standard(
					fsc_out_pdbqt,
					&DSDP_sort.selected_fsc_crd[(size_t)i * (flexsc[0].atom_numbers)],
					move_vec, i, score_out_i); // FIXME this maybe rightS

				copy_pdbqt_ligand.Append_Frame_To_Opened_pdbqt_standard(
					ligand_out_pdbqt,
					&DSDP_sort.selected_lig_crd[(size_t)i * (ligand[0].atom_numbers)],
					move_vec, i, score_out_i);
				// 经过修改的copy_pdbqt，识别HETATM
				fprintf(out_list, "%s %f\n", ligand_name, score_out_i);
			}
			fprintf(out_list, "task time %lf s\n", omp_get_wtime() - task_time_begin);
			fclose(ligand_out_pdbqt);
			fclose(fsc_out_pdbqt);

			printf("done.\n");
			printf("task time %lf s\n\n", omp_get_wtime() - task_time_begin);
		}
		// if err in ligand task?
		catch (const std::exception &e)
		{
			std::cerr << e.what() << '\n';
			printf("%s failed. Skipped for this task.\n", ligand_name);
		}

	} // task loop

	fclose(out_list); // this is opened in main
	time_begin = omp_get_wtime() - time_begin;
	printf("Total time %lf s\n", time_begin);
	return 0;
}

#ifndef COMMON_CUH
#define COMMON_CUH
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <string>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 0~(ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX-1)，且第0个记录的是该格子目前记录的原子数目+1，因此实际只能容纳ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX-1个
// 建议计数为for(int i=1;i<atom_serial[0];i=i+1)，实际含有原子数为atom_serial[0]-1
#define ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX 64

struct NEIGHBOR_GRID_BUCKET
{
	int atom_serial[ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX];
};
struct INT_FLOAT
{
	int id;
	float energy;
};
bool cmp(INT_FLOAT &a, INT_FLOAT &b);

struct UINT2
{
	unsigned int a;
	unsigned int b;
};
struct VECTOR
{
	float x;
	float y;
	float z;
};
struct VECTOR_INT
{
	float x;
	float y;
	float z;
	int type;
};
struct INT_VECTOR
{
	int int_x;
	int int_y;
	int int_z;
};

// 二面角转动的节点信息结构
struct NODE
{
	float matrix[9];		// 记录绕轴n的转动theta角度的矩阵
	int root_atom_serial;	// 矢量a时刻从该atom crd中获得
	int branch_atom_serial; // 矢量n时刻从该atom crd和a中计算得到
	VECTOR a0, n0, a, n;	// 初始的轴位置a0和轴指向n0，当前的a，n
	int last_node_serial;	// 上一个节点的序号（父节点）
};
struct Residue
{
	std::string res_name;
	int res_id;
};
// Vina 力场参数
#define k_gauss1 -0.035579f
#define k_gauss1_2 4.f

#define k_gauss2 -0.005156f
#define k_gauss2_2 0.25f
#define k_gauss2_c 3.f

#define k_repulsion +0.840245f

#define k_hydrophobic -0.035069f
#define k_hydrophobic_a 0.5f
#define k_hydrophobic_b 1.5f
#define k_hydrophobic_ua 1.f
#define k_hydrophobic_ub 0.f

#define k_h_bond -0.587439f
#define k_h_bond_a -0.7f
#define k_h_bond_b 0.f
#define k_h_bond_ua 1.f
#define k_h_bond_ub 0.f
struct VINA_ATOM
{
	VECTOR crd;			// 原子坐标
	int atom_type;		// 原子类型，默认为18种
	float radius;		// 原子半径，计算Vina打分时需要
	int is_hydrophobic; // =1 代表疏水原子， =0代表亲水原子
	int is_donor;		// =1代表供体
	int is_acceptor;	// =1代表受体

	// float charge;//如有需要，可自行添加额外力场函数，如静电相互作用
};

FILE *fopen_safely(const char *file_name, const char *mode);
float real_distance(VECTOR &crd, VECTOR &crd2);
// 返回三个欧拉角（绕固定的x、y、z轴旋转）
// 利用此返回角度进行转动，可保证转动构象与参考构象无关（统计意义上）
VECTOR uniform_rand_Euler_angles();
VECTOR uniform_rand_Euler_angles_old();
VECTOR uniform_rand_Euler_angles_range(float range);
// 本程序的所有rmsd如无强调都是无平移和转动自由度的
// a,b是两套要比较的原子坐标
float calcualte_heavy_atom_rmsd(const int atom_numbers, const VECTOR *a, const VECTOR *b, const int *atomic_number);

// 主要针对PDBQT格式的最后一列原子名称返回原子序数
int Get_Atomic_Number_From_PDBQT_Atom_Name(const char *atom_name);

void Read_Atom_Line_In_PDBQT(const char *line, std::vector<VECTOR> &crd, std::vector<float> &charge, std::vector<int> &atom_type);

// vina的近邻表需要剔除这些内容：自己、1-2bond相连，由1-2产生的1-3和1-4、所有氢元素、属于同一个刚体内部的
// 注意，这里的同一个刚体内部同时也包括了实际转动无效的-O-H,和-N(-H)-H等结构
// neighbor_list是记录的剔除后剩余的需要计算的分子内相互作用pair（三角阵，但atom_numbers*atom_numbers，具体存储见实现）
// atom_node_serial用于剔除属于同一个刚体内部的相互作用
void Build_Inner_Neighbor_List(const int atom_numbers, int *neighbor_list, std::vector<VECTOR> &initial_crd, std::vector<int> &atomic_number,
							   std::vector<int> &atom_node_serial);

// 根据pdbqt中的atom_type和初始坐标，按照大致vina逻辑生成vina_atom用于计算力场
int Build_Vina_Atom(VINA_ATOM *vina_atom, std::vector<int> &atom_type, std::vector<VECTOR> &initial_crd, std::vector<int> &atomic_number);

// 两个原子的vina打分，返回{力的绝对值,能量}
float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b);
float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b, const float dr); // 在计算a,b相互作用时用外部给定的距离dr
// Debug
void print_vec(VECTOR v, const char *name);

double v_inner_product(VECTOR a, VECTOR b);
VECTOR v_plus(VECTOR a, VECTOR b);
VECTOR v_minus(VECTOR a, VECTOR b); // return a-b
VECTOR v_times(VECTOR a, float w);
// Dihedral
float calc_dihedral(VECTOR va, VECTOR vb, VECTOR vc);
bool is_large_rotation(float x, float y, float z, float gate);
void matrix_multiply_vector(VECTOR *__restrict__ c, const float *__restrict__ a, const VECTOR *__restrict__ b);
float random_2pi();
float random_2A();
// input/output files of a single task
struct TaskIO
{
	std::string lig_in;
	std::string lig_out;
	std::string fsc_out;
};
int Read_list_from_file(std::string file_name,
						std::vector<TaskIO> *task_io_list);
#endif // COMMON_CUH

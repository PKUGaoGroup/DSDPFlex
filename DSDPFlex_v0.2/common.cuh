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

// 0~(ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX-1)���ҵ�0����¼���Ǹø���Ŀǰ��¼��ԭ����Ŀ+1�����ʵ��ֻ������ATOM_NUMBES_IN_NEIGHBOR_GRID_BUCKET_MAX-1��
// �������Ϊfor(int i=1;i<atom_serial[0];i=i+1)��ʵ�ʺ���ԭ����Ϊatom_serial[0]-1
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

// �����ת���Ľڵ���Ϣ�ṹ
struct NODE
{
	float matrix[9];		// ��¼����n��ת��theta�Ƕȵľ���
	int root_atom_serial;	// ʸ��aʱ�̴Ӹ�atom crd�л��
	int branch_atom_serial; // ʸ��nʱ�̴Ӹ�atom crd��a�м���õ�
	VECTOR a0, n0, a, n;	// ��ʼ����λ��a0����ָ��n0����ǰ��a��n
	int last_node_serial;	// ��һ���ڵ����ţ����ڵ㣩
};
struct Residue
{
	std::string res_name;
	int res_id;
};
// Vina ��������
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
	VECTOR crd;			// ԭ������
	int atom_type;		// ԭ�����ͣ�Ĭ��Ϊ18��
	float radius;		// ԭ�Ӱ뾶������Vina���ʱ��Ҫ
	int is_hydrophobic; // =1 ������ˮԭ�ӣ� =0������ˮԭ��
	int is_donor;		// =1��������
	int is_acceptor;	// =1��������

	// float charge;//������Ҫ�����������Ӷ��������������羲���໥����
};

FILE *fopen_safely(const char *file_name, const char *mode);
float real_distance(VECTOR &crd, VECTOR &crd2);
// ��������ŷ���ǣ��ƹ̶���x��y��z����ת��
// ���ô˷��ؽǶȽ���ת�����ɱ�֤ת��������ο������޹أ�ͳ�������ϣ�
VECTOR uniform_rand_Euler_angles();
VECTOR uniform_rand_Euler_angles_old();
VECTOR uniform_rand_Euler_angles_range(float range);
// �����������rmsd����ǿ��������ƽ�ƺ�ת�����ɶȵ�
// a,b������Ҫ�Ƚϵ�ԭ������
float calcualte_heavy_atom_rmsd(const int atom_numbers, const VECTOR *a, const VECTOR *b, const int *atomic_number);

// ��Ҫ���PDBQT��ʽ�����һ��ԭ�����Ʒ���ԭ������
int Get_Atomic_Number_From_PDBQT_Atom_Name(const char *atom_name);

void Read_Atom_Line_In_PDBQT(const char *line, std::vector<VECTOR> &crd, std::vector<float> &charge, std::vector<int> &atom_type);

// vina�Ľ��ڱ���Ҫ�޳���Щ���ݣ��Լ���1-2bond��������1-2������1-3��1-4��������Ԫ�ء�����ͬһ�������ڲ���
// ע�⣬�����ͬһ�������ڲ�ͬʱҲ������ʵ��ת����Ч��-O-H,��-N(-H)-H�Ƚṹ
// neighbor_list�Ǽ�¼���޳���ʣ�����Ҫ����ķ������໥����pair�������󣬵�atom_numbers*atom_numbers������洢��ʵ�֣�
// atom_node_serial�����޳�����ͬһ�������ڲ����໥����
void Build_Inner_Neighbor_List(const int atom_numbers, int *neighbor_list, std::vector<VECTOR> &initial_crd, std::vector<int> &atomic_number,
							   std::vector<int> &atom_node_serial);

// ����pdbqt�е�atom_type�ͳ�ʼ���꣬���մ���vina�߼�����vina_atom���ڼ�������
int Build_Vina_Atom(VINA_ATOM *vina_atom, std::vector<int> &atom_type, std::vector<VECTOR> &initial_crd, std::vector<int> &atomic_number);

// ����ԭ�ӵ�vina��֣�����{���ľ���ֵ,����}
float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b);
float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b, const float dr); // �ڼ���a,b�໥����ʱ���ⲿ�����ľ���dr
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
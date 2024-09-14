#ifndef KERNEL_CUH
#define KERNEL_CUH
#include "common.cuh"
#include "Partial_Rigid_Small_Molecule.cuh"
#include "Flexible_Side_Chain.cuh"

#define HYDROGEN_ATOM_TYPE_SERIAL 17
#define MAX_OPTIMIZE_STEPS 100
#define BIG_ENERGY 1000.f
#define CONVERGENCE_CUTOFF 0.0001f
#define MAX_OPTIM_TURNS 25
#define MAX_OPTIM_STEP_PER_TURN 4

struct Optim_shared
{
	const long long int *protein_mesh;
	const float box_border_strenth;
	const VECTOR box_min;
	const VECTOR box_max;
	const VECTOR protein_mesh_grid_length_inverse;
};

struct Optim_struct
{
	const int atom_numbers;
	const int *inner_interaction_list;
	const float cutoff;
	const int *atom_to_node_serial;
	const VECTOR *ref_crd;
	VINA_ATOM *vina_atom;
	VECTOR *frc;
	float *energy;

	const int u_freedom;
	float *u_crd;
	float *last_u_crd;
	float *dU_du_crd;
	float *last_dU_du_crd;
	const int node_numbers;
	NODE *node;
};
struct E_results
{
	float lig_receptor_energy;
	float lig_lig_energy;
	float lig_fsc_energy;
	float fsc_receptor_energy;
	float fsc_fsc_energy;
};
// inner_interaction_list是这样一种结构，用于记录每个原子需要考虑计算同一分子内两体作用的列表
// 为方便分配，实际上inner_interaction_list是个atom_numbers*atom_numbers的矩阵
// 但每个inner_interaction_list[i*atom_numbers]代表i号原子要考虑两体作用的原子数（存储的考虑编号总是大于i）
// 且为了保证效率，要求每一行inner_interaction_list[i*atom_numbers]后面的原子序号都是排了序的。
// frc和energy都会在该kernel里清零重加，因此无需保证输入的frc和energy初始化
// 为保持一致性，原子crd坐标均采用VECTOR_INT，int记录的是原子种类。
__global__ void
Calculate_Energy_And_Grad_Device(
	const int atom_numbers, const int *inner_interaction_list, const float cutoff,
	const VECTOR_INT *vina_atom, VECTOR *frc, float *energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse);

// 算力相关的变量可参考上面的kernel函数
// ref_crd是对应u_crd、node的参考坐标，所有vina_atom内的坐标均由它生成
// atom_to_node_serial
__global__ void Optimize_Structure_Device(
	const int atom_numbers, const int *inner_interaction_list, const float cutoff,
	const int *atom_to_node_serial,
	const VECTOR *ref_crd, VECTOR_INT *vina_atom, VECTOR *frc, float *energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float *u_crd, float *last_u_crd, float *dU_du_crd, float *last_dU_du_crd,
	const int node_numbers, NODE *node);

// 算力相关的变量可参考上面的kernel函数
// ref_crd是对应u_crd、node的参考坐标，所有vina_atom内的坐标均由它生成
// atom_to_node_serial
__global__ void Optimize_Structure_BB2_Device(
	const int atom_numbers, const int *inner_interaction_list, const float cutoff,
	const int *atom_to_node_serial,
	const VECTOR *ref_crd, VECTOR_INT *vina_atom, VECTOR *frc, float *energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float *u_crd, float *last_u_crd, float *dU_du_crd, float *last_dU_du_crd,
	const int node_numbers, NODE *node);

// 对pair作用不使用插值表，直接进行计算
// 经测试，比上面的使用pair作用插值的要快很多，重点并不是在于减少计算量（反而增加了极大的计算量），而是在于避免每个kernel都读取公共的pair_potential
__global__ void Optimize_Structure_BB2_Direct_Pair_Device_modeLIG(
	const int atom_numbers, const int *inner_interaction_list, const float cutoff,
	const int *atom_to_node_serial,
	const VECTOR *ref_crd, VINA_ATOM *vina_atom, VECTOR *frc, float *energy,
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max,
	const VECTOR transbox_min, const VECTOR transbox_max,
	const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float *u_crd, float *last_u_crd, float *dU_du_crd, float *last_dU_du_crd,
	const int node_numbers, NODE *node,
	const int fsc_atom_numbers, VINA_ATOM *fsc_vina_atom);

__global__ void Optimize_Structure_BB2_Direct_Pair_Device_modeSC(
	const int atom_numbers, const int *inner_interaction_list, const float cutoff,
	const int *atom_to_node_serial,
	const VECTOR *ref_crd, VINA_ATOM *vina_atom, VECTOR *frc, float *energy, float *inter_energy,
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float *u_crd, float *last_u_crd, float *dU_du_crd, float *last_dU_du_crd,
	const int node_numbers, NODE *node,
	const int lig_atom_numbers, VINA_ATOM *lig_vina_atom);

__global__ void Update_Structure_LIG(
	const int atom_numbers, const int *atom_to_node_serial,
	const VECTOR *ref_crd, VINA_ATOM *vina_atom,
	const int u_freedom, float *u_crd, float *last_u_crd,
	const int node_numbers, NODE *node);

__global__ void Update_Structure_SC(
	const int atom_numbers, const int *atom_to_node_serial,
	const VECTOR *ref_crd, VINA_ATOM *vina_atom,
	const int u_freedom, float *u_crd, float *last_u_crd,
	const int node_numbers, NODE *node);

__global__ void Optimize_All_Structure_BB2_Direct_Pair_Device(
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR protein_mesh_grid_length_inverse,
	const VECTOR box_min, const VECTOR box_max,
	const VECTOR transbox_min, const VECTOR transbox_max,
	const float cutoff,
	PARTIAL_RIGID_SMALL_MOLECULE::GPU ligand,
	FLEXIBLE_SIDE_CHAIN::GPU flex_chains,
	float *l_energy, float *f_energy, float *inter_energy,
	const float flex_ratio);

__global__ void New_Bootstrap_Optimization_Kernel(
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR protein_mesh_grid_length_inverse,
	const VECTOR box_min, const VECTOR box_max,
	const VECTOR transbox_min, const VECTOR transbox_max,
	const float cutoff,
	PARTIAL_RIGID_SMALL_MOLECULE::GPU ligand,
	FLEXIBLE_SIDE_CHAIN::GPU flex_chains,
	float *l_energy, float *f_energy, float *inter_energy,
	const float flex_ratio);
__device__ void ligand_ref_crd_update(PARTIAL_RIGID_SMALL_MOLECULE::GPU *ligand_gpu);

// Update ref_crd and NODE just in place
__global__ void update_structure_ref_inplace(
	const int atom_numbers, const int *atom_to_node_serial,
	VECTOR *ref_crd, VINA_ATOM *vina_atom,
	const int u_freedom, float *u_crd, float *last_u_crd,
	const int node_numbers, NODE *node);
#endif // KERNEL_CUH

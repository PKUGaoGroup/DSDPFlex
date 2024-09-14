#include "Kernel.cuh"
__device__ __host__ static void Matrix_Multiply_Vector(VECTOR *__restrict__ c, const float *__restrict__ a, const VECTOR *__restrict__ b)
{
	c[0].x = a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z;
	c[0].y = a[3] * b[0].x + a[4] * b[0].y + a[5] * b[0].z;
	c[0].z = a[6] * b[0].x + a[7] * b[0].y + a[8] * b[0].z;
}

// inner_interaction_list是这样一种结构，用于记录每个原子需要考虑计算同一分子内两体作用的列表
// 为方便分配，实际上inner_interaction_list是个atom_numbers*atom_numbers的矩阵
// 但每个inner_interaction_list[i*atom_numbers]代表i号原子要考虑两体作用的原子数（存储的考虑编号总是大于i）
// 且为了保证效率，要求每一行inner_interaction_list[i*atom_numbers]后面的原子序号都是排了序的。
// frc和energy都会在该kernel里清零重加，因此无需保证输入的frc和energy初始化
// 为保持一致性，原子crd坐标均采用VECTOR_INT，int记录的是原子种类。
__global__ void Calculate_Energy_And_Grad_Device(
	const int atom_numbers, const int *inner_interaction_list, const float cutoff,
	const VECTOR_INT *vina_atom, VECTOR *frc, float *energy,
	const float pair_potential_grid_length_inverse, const cudaTextureObject_t pair_potential,
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse)
{
	if (threadIdx.x == 0)
	{
		energy[0] = 0.f;
	}
	float total_energy_in_thread = 0.f;
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
	{
		VECTOR_INT atom_i = vina_atom[i];
		VECTOR force_i = {0.f, 0.f, 0.f};
		VECTOR dr;
		if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
		{
			// box interaction
			dr.x = fdimf(box_min.x, atom_i.x); // 如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
			dr.y = fdimf(box_min.y, atom_i.y);
			dr.z = fdimf(box_min.z, atom_i.z);
			force_i.x += box_border_strenth * dr.x;
			force_i.y += box_border_strenth * dr.y;
			force_i.z += box_border_strenth * dr.z;
			total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

			dr.x = fdimf(atom_i.x, box_max.x);
			dr.y = fdimf(atom_i.y, box_max.y);
			dr.z = fdimf(atom_i.z, box_max.z);
			force_i.x -= box_border_strenth * dr.x;
			force_i.y -= box_border_strenth * dr.y;
			force_i.z -= box_border_strenth * dr.z;
			total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

			// protein interaction
			VECTOR serial; // 在蛋白插值网格中的格点坐标
			serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
			serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
			serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
			float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
			total_energy_in_thread += ans.w;
			force_i.x += ans.x;
			force_i.y += ans.y;
			force_i.z += ans.z;
		}
		frc[i] = force_i; // 该kernel不会在输入的frc上累加
	}
	__syncthreads(); // 同步，以保证后面两两作用加力时已经全部走过这一步，同时保证能量也成功清零

	VECTOR_INT atom_i, atom_j;
	VECTOR force_i, temp_force;
	VECTOR dr;
	float rij, dd, dd_, frc_abs, rij_inverse;
	float4 ans;
	int inner_list_start;
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
	{
		atom_i = vina_atom[i];
		force_i = {0.f, 0.f, 0.f};
		inner_list_start = i * atom_numbers;
		int inner_numbers = inner_interaction_list[inner_list_start];
		for (int k = 1; k <= inner_numbers; k = k + 1)
		{
			int j = inner_interaction_list[inner_list_start + k];
			atom_j = vina_atom[j];
			dr = {atom_i.x - atom_j.x, atom_i.y - atom_j.y, atom_i.z - atom_j.z};
			rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
			if (rij < cutoff)
			{
				rij_inverse = 1.f / (rij + 10.e-6f);
				rij *= pair_potential_grid_length_inverse; // 变为两体作用插值表的格点坐标
				dd = rij - floor(rij);
				dd_ = 1.f - dd;
				ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type); // 不自带插值，可考虑后续替换为多个独立的一维表，类似protein mesh一样
				frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;								  // 好像发现距离矢量不归一化的对接效果会更好，即分子内的两体作用的力根据距离线性扩大（相比严格正确的力，但由于整体是高斯衰减的，所以只相当于平衡距离被拖远了一点）
				total_energy_in_thread += 0.5f * (dd_ * ans.y + dd * ans.w);					  // 加两次除以2

				temp_force.x = frc_abs * dr.x;
				temp_force.y = frc_abs * dr.y;
				temp_force.z = frc_abs * dr.z;
				force_i.x += temp_force.x;
				force_i.y += temp_force.y;
				force_i.z += temp_force.z;
				atomicAdd(&frc[j].x, -temp_force.x);
				atomicAdd(&frc[j].y, -temp_force.y);
				atomicAdd(&frc[j].z, -temp_force.z);
			}
		}
		atomicAdd(&frc[i].x, force_i.x);
		atomicAdd(&frc[i].y, force_i.y);
		atomicAdd(&frc[i].z, force_i.z);
	}
	atomicAdd(&energy[0], total_energy_in_thread);
}

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
	const int node_numbers, NODE *node)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[19];
	float *rot_matrix = &shared_data[2];
	float *alpha = &shared_data[17];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;		 // 临时能量项
		shared_data[1] = BIG_ENERGY; // 临时能量项
									 // shared_data[2]...shared_data[10]//整体转动矩阵
									 // shared_data[11] cos_b 均对应欧拉转动角
									 // shared_data[12] sin_b
									 // shared_data[13] cos_a
									 // shared_data[14] sin_a
									 // shared_data[15] cacb
									 // shared_data[16] cbsa
									 // shared_data[17] = 0.f;//BB优化用的alpha
									 // shared_data[18] = 0.f;
	}

	// 进入主循环前的基本初始化
	for (int i = 0; i < u_freedom; i = i + 1)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		last_u_crd[i] = u_crd[i];
	}

	// 进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		// 在当前广义坐标下更新各转动矩阵
		for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				node[i].matrix[0] = temp_matrix_1[0];
				node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				node[i].matrix[4] = temp_matrix_1[4];
				node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
				sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
				sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

				rot_matrix[0] = cos_b * cos_c;
				rot_matrix[1] = cos_b * sin_c;
				rot_matrix[2] = -sin_b;
				rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				rot_matrix[5] = cos_b * sin_a;
				rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				rot_matrix[8] = cos_a * cos_b;

				shared_data[11] = cos_b;
				shared_data[12] = sin_b;
				shared_data[13] = cos_a;
				shared_data[14] = sin_a;
				shared_data[15] = rot_matrix[8]; // cacb
				shared_data[16] = rot_matrix[5]; // cbsa
			}
		}
		__syncthreads();

		// 由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].x = temp_crd2.x + u_crd[u_freedom - 6] + center.x; // 整体平移在最后加上
			vina_atom[i].y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		// 由当前坐标更新node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
		for (int node_id = threadIdx.x; node_id < node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = {vina_atom[node[node_id].root_atom_serial].x, vina_atom[node[node_id].root_atom_serial].y, vina_atom[node[node_id].root_atom_serial].z};
			tempn = {vina_atom[node[node_id].branch_atom_serial].x, vina_atom[node[node_id].branch_atom_serial].y, vina_atom[node[node_id].branch_atom_serial].z};
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			node[node_id].n = tempn;
			node[node_id].a = tempa;
		}
		//__syncthreads();//这里实际不需要同步

		// 计算原子力和总能量
		float total_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR_INT atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VECTOR_INT atom_i = vina_atom[i];
			VECTOR force_i = {0.f, 0.f, 0.f};
			VECTOR dr;
			if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
			{
				// box interaction
				dr.x = fdimf(box_min.x, atom_i.x); // 如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
				dr.y = fdimf(box_min.y, atom_i.y);
				dr.z = fdimf(box_min.z, atom_i.z);
				force_i.x += box_border_strenth * dr.x;
				force_i.y += box_border_strenth * dr.y;
				force_i.z += box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				dr.x = fdimf(atom_i.x, box_max.x);
				dr.y = fdimf(atom_i.y, box_max.y);
				dr.z = fdimf(atom_i.z, box_max.z);
				force_i.x -= box_border_strenth * dr.x;
				force_i.y -= box_border_strenth * dr.y;
				force_i.z -= box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				// protein interaction
				VECTOR serial; // 在蛋白插值网格中的格点坐标
				serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
				total_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;
			}
			inner_list_start = i * atom_numbers;
			int inner_numbers = inner_interaction_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = inner_interaction_list[inner_list_start + k];
				atom_j = vina_atom[j];
				dr = {atom_i.x - atom_j.x, atom_i.y - atom_j.y, atom_i.z - atom_j.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
				if (rij < cutoff)
				{
					rij_inverse = 1.f / (rij + 10.e-6f);
					rij *= pair_potential_grid_length_inverse; // 变为两体作用插值表的格点坐标
					dd = rij - floor(rij);
					dd_ = 1.f - dd;
					ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type); // 不自带插值，可考虑后续替换为多个独立的一维表，类似protein mesh一样
					frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;								  // 好像发现距离矢量不归一化的对接效果会更好，即分子内的两体作用的力根据距离线性扩大（相比严格正确的力，但由于整体是高斯衰减的，所以只相当于平衡距离被拖远了一点）
					total_energy_in_thread += (dd_ * ans.y + dd * ans.w);							  // 如果inner list是不重复计算pair作用的，则不需要乘0.5f

					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&frc[j].x, -temp_force.x);
					atomicAdd(&frc[j].y, -temp_force.y);
					atomicAdd(&frc[j].z, -temp_force.z);
				}
			}
			atomicAdd(&frc[i].x, force_i.x);
			atomicAdd(&frc[i].y, force_i.y);
			atomicAdd(&frc[i].z, force_i.z);
		}
		atomicAdd(&shared_data[0], total_energy_in_thread);
		__syncthreads(); // 能量加和完全，且梯度以及node的叉乘相关信息完全

		// 提前退出优化
		if (fabsf(shared_data[0] - shared_data[1]) < CONVERGENCE_CUTOFF)
		{
			if (threadIdx.x == 0)
			{
				energy[0] = shared_data[0];
			}
			break;
		}
		if (threadIdx.x == 0)
		{
			shared_data[1] = shared_data[0];
			shared_data[0] = 0.f;
			alpha[0] = 0.f;
			alpha[1] = 0.f;
		}

		// 计算广义力
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = {vina_atom[0].x, vina_atom[0].y, vina_atom[0].z};
			VECTOR temp_crd2 = {vina_atom[i].x, vina_atom[i].y, vina_atom[i].z};
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&dU_du_crd[u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&dU_du_crd[u_freedom - 2], (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]));
			atomicAdd(&dU_du_crd[u_freedom - 3], (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12])));

			atomicAdd(&dU_du_crd[u_freedom - 6], temp_frc.x);
			atomicAdd(&dU_du_crd[u_freedom - 5], temp_frc.y);
			atomicAdd(&dU_du_crd[u_freedom - 4], temp_frc.z);

			int current_node_id = atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - node[current_node_id].a.z;
				rot_axis = node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		// 进行BB优化更新(暂时未区分整体转动、平动和二面角自由度的各自优化)
		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float s = u_crd[i] - last_u_crd[i];
			float y = dU_du_crd[i] - last_dU_du_crd[i];
			atomicAdd(&alpha[0], y * s);
			atomicAdd(&alpha[1], y * y);
			last_u_crd[i] = u_crd[i];
			last_dU_du_crd[i] = dU_du_crd[i];
		}
		__syncthreads();

		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float temp_alpha = fabsf(alpha[0]) / fmaxf(alpha[1], 1.e-6f);
			float du = temp_alpha * dU_du_crd[i];
			dU_du_crd[i] = 0.f;
			du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			u_crd[i] += du;
		}
	}
}

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
	const int node_numbers, NODE *node)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[23];
	float *rot_matrix = &shared_data[2];
	float *alpha1 = &shared_data[17];
	float *alpha2 = &shared_data[19];
	float *alpha3 = &shared_data[21];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;		 // 临时能量项
		shared_data[1] = BIG_ENERGY; // 临时能量项
									 // shared_data[2]...shared_data[10]//整体转动矩阵
									 // shared_data[11] cos_b 均对应欧拉转动角
									 // shared_data[12] sin_b
									 // shared_data[13] cos_a
									 // shared_data[14] sin_a
									 // shared_data[15] cacb
									 // shared_data[16] cbsa
									 // shared_data[17] = 0.f;//BB优化用的alpha
									 // shared_data[18] = 0.f;
									 // shared_data[19] = 0.f;//BB优化用的alpha
									 // shared_data[20] = 0.f;
									 // shared_data[21] = 0.f;//BB优化用的alpha
									 // shared_data[22] = 0.f;
	}

	// 进入主循环前的基本初始化
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		if (i < u_freedom - 6 || u_freedom - 3 < i)
		{
			// u_crd[i] = 0.f;
		}
		last_u_crd[i] = u_crd[i];
	}

	// 进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		// 在当前广义坐标下更新各转动矩阵
		for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				node[i].matrix[0] = temp_matrix_1[0];
				node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				node[i].matrix[4] = temp_matrix_1[4];
				node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
				sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
				sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

				rot_matrix[0] = cos_b * cos_c;
				rot_matrix[1] = cos_b * sin_c;
				rot_matrix[2] = -sin_b;
				rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				rot_matrix[5] = cos_b * sin_a;
				rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				rot_matrix[8] = cos_a * cos_b;

				shared_data[11] = cos_b;
				shared_data[12] = sin_b;
				shared_data[13] = cos_a;
				shared_data[14] = sin_a;
				shared_data[15] = rot_matrix[8]; // cacb
				shared_data[16] = rot_matrix[5]; // cbsa
			}
		}
		__syncthreads();

		// 由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}
			temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].x = temp_crd2.x + u_crd[u_freedom - 6] + center.x; // 整体平移在最后加上
			vina_atom[i].y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		// 由当前坐标更新node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
		for (int node_id = threadIdx.x; node_id < node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = {vina_atom[node[node_id].root_atom_serial].x, vina_atom[node[node_id].root_atom_serial].y, vina_atom[node[node_id].root_atom_serial].z};
			tempn = {vina_atom[node[node_id].branch_atom_serial].x, vina_atom[node[node_id].branch_atom_serial].y, vina_atom[node[node_id].branch_atom_serial].z};
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			node[node_id].n = tempn;
			node[node_id].a = tempa;
		}
		//__syncthreads();//这里实际不需要同步

		// 计算原子力和总能量
		float total_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR_INT atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VECTOR_INT atom_i = vina_atom[i];
			VECTOR force_i = {0.f, 0.f, 0.f};
			VECTOR dr;
			if (atom_i.type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
			{
				// box interaction
				dr.x = fdimf(box_min.x, atom_i.x); // 如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
				dr.y = fdimf(box_min.y, atom_i.y);
				dr.z = fdimf(box_min.z, atom_i.z);
				force_i.x += box_border_strenth * dr.x;
				force_i.y += box_border_strenth * dr.y;
				force_i.z += box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				dr.x = fdimf(atom_i.x, box_max.x);
				dr.y = fdimf(atom_i.y, box_max.y);
				dr.z = fdimf(atom_i.z, box_max.z);
				force_i.x -= box_border_strenth * dr.x;
				force_i.y -= box_border_strenth * dr.y;
				force_i.z -= box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				// protein interaction
				VECTOR serial; // 在蛋白插值网格中的格点坐标
				serial.x = (atom_i.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				float4 ans = tex3D<float4>(protein_mesh[atom_i.type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
				// float4 ans = { 0.f,0.f,0.f };
				total_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;
			}
			inner_list_start = i * atom_numbers;
			int inner_numbers = inner_interaction_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = inner_interaction_list[inner_list_start + k];
				atom_j = vina_atom[j];
				dr = {atom_i.x - atom_j.x, atom_i.y - atom_j.y, atom_i.z - atom_j.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
				if (rij < cutoff)
				{
					rij_inverse = 1.f / (rij + 10.e-6f);
					rij *= pair_potential_grid_length_inverse; // 变为两体作用插值表的格点坐标
					dd = rij - floor(rij);
					dd_ = 1.f - dd;
					ans = tex3D<float4>(pair_potential, rij, (float)atom_i.type, (float)atom_j.type); // 不自带插值，可考虑后续替换为多个独立的一维表，类似protein mesh一样
					// ans = { 0.f,0.f,0.f,0.f };
					frc_abs = (dd_ * ans.x + dd * ans.z) * rij_inverse;	  // 好像发现距离矢量不归一化的对接效果会更好，即分子内的两体作用的力根据距离线性扩大（相比严格正确的力，但由于整体是高斯衰减的，所以只相当于平衡距离被拖远了一点）
					total_energy_in_thread += (dd_ * ans.y + dd * ans.w); // 如果inner list是不重复计算pair作用的，则不需要乘0.5f

					// LJ test
					/*float r_2 = rij_inverse * rij_inverse;
					float r_4 = r_2 * r_2;
					float r_6 = r_4 * r_2;
					frc_abs = -1.f*(-12.f * 9.4429323e+05f * r_6 + 6.f * 8.0132353e+02f) * r_6 * r_2;
					total_energy_in_thread += 1.f * (9.4429323e+05f * r_6 - 8.0132353e+02f) * r_6;*/

					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&frc[j].x, -temp_force.x);
					atomicAdd(&frc[j].y, -temp_force.y);
					atomicAdd(&frc[j].z, -temp_force.z);
				}
			}
			atomicAdd(&frc[i].x, force_i.x);
			atomicAdd(&frc[i].y, force_i.y);
			atomicAdd(&frc[i].z, force_i.z);
		}
		atomicAdd(&shared_data[0], total_energy_in_thread);
		__syncthreads(); // 能量加和完全，且梯度以及node的叉乘相关信息完全

		// 提前退出优化
		if (fabsf(shared_data[0] - shared_data[1]) < CONVERGENCE_CUTOFF)
		{
			if (threadIdx.x == 0)
			{
				energy[0] = shared_data[0];
			}
			// break;
		}
		if (threadIdx.x == 0)
		{
			shared_data[1] = shared_data[0];
			shared_data[0] = 0.f;
			alpha1[0] = 0.f;
			alpha1[1] = 0.f;
			alpha2[0] = 0.f;
			alpha2[1] = 0.f;
			alpha3[0] = 0.f;
			alpha3[1] = 0.f;
		}

		// 计算广义力
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = {vina_atom[0].x, vina_atom[0].y, vina_atom[0].z};
			VECTOR temp_crd2 = {vina_atom[i].x, vina_atom[i].y, vina_atom[i].z};
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&dU_du_crd[u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&dU_du_crd[u_freedom - 2], (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]));
			atomicAdd(&dU_du_crd[u_freedom - 3], (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12])));

			atomicAdd(&dU_du_crd[u_freedom - 6], temp_frc.x);
			atomicAdd(&dU_du_crd[u_freedom - 5], temp_frc.y);
			atomicAdd(&dU_du_crd[u_freedom - 4], temp_frc.z);

			int current_node_id = atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - node[current_node_id].a.z;
				rot_axis = node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		// 进行BB优化更新(暂时未区分整体转动、平动和二面角自由度的各自优化)
		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float s = u_crd[i] - last_u_crd[i];
			float y = dU_du_crd[i] - last_dU_du_crd[i];
			last_u_crd[i] = u_crd[i];
			last_dU_du_crd[i] = dU_du_crd[i];
			if (i < u_freedom - 6)
			{
				atomicAdd(&alpha1[0], y * s);
				atomicAdd(&alpha1[1], y * y);
			}
			else if (i < u_freedom - 3)
			{
				atomicAdd(&alpha2[0], y * s);
				atomicAdd(&alpha2[1], y * y);
			}
			else
			{
				atomicAdd(&alpha3[0], y * s);
				atomicAdd(&alpha3[1], y * y);
			}
		}
		__syncthreads();

		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float du;
			if (i < u_freedom - 6)
			{
				float temp_alpha = fabsf(alpha1[0]) / fmaxf(alpha1[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			}
			else if (i < u_freedom - 3)
			{
				float temp_alpha = fabsf(alpha2[0]) / fmaxf(alpha2[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 1.f / 10000.f), du);
			}
			else
			{
				float temp_alpha = fabsf(alpha3[0]) / fmaxf(alpha3[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 2.f * 3.141592654f / 100000.f), du);
			}
			dU_du_crd[i] = 0.f;
			u_crd[i] += du;
		}
		__syncthreads();
	}
}

// 对pair作用不使用插值表，直接进行计算
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
	const int fsc_atom_numbers, VINA_ATOM *fsc_vina_atom)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[23];
	float *rot_matrix = &shared_data[2];
	float *alpha1 = &shared_data[17];
	float *alpha2 = &shared_data[19];
	float *alpha3 = &shared_data[21];
	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;		 // 临时能量项
		shared_data[1] = BIG_ENERGY; // 临时能量项
	}

	// 进入主循环前的基本初始化
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		last_u_crd[i] = u_crd[i];
	}

	// 进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		// 在当前广义坐标下更新各转动矩阵
		for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				node[i].matrix[0] = temp_matrix_1[0];
				node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				node[i].matrix[4] = temp_matrix_1[4];
				node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
				sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
				sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

				rot_matrix[0] = cos_b * cos_c;
				rot_matrix[1] = cos_b * sin_c;
				rot_matrix[2] = -sin_b;
				rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				rot_matrix[5] = cos_b * sin_a;
				rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				rot_matrix[8] = cos_a * cos_b;

				shared_data[11] = cos_b;
				shared_data[12] = sin_b;
				shared_data[13] = cos_a;
				shared_data[14] = sin_a;
				shared_data[15] = rot_matrix[8]; // cacb
				shared_data[16] = rot_matrix[5]; // cbsa
			}
		}
		__syncthreads();

		// 由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}

			temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			vina_atom[i].crd.x = temp_crd2.x + u_crd[u_freedom - 6] + center.x; // 整体平移在最后加上
			vina_atom[i].crd.y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
			vina_atom[i].crd.z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
		}
		__syncthreads();

		// 由当前坐标更新node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
		for (int node_id = threadIdx.x; node_id < node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = {vina_atom[node[node_id].root_atom_serial].crd.x, vina_atom[node[node_id].root_atom_serial].crd.y, vina_atom[node[node_id].root_atom_serial].crd.z};
			tempn = {vina_atom[node[node_id].branch_atom_serial].crd.x, vina_atom[node[node_id].branch_atom_serial].crd.y, vina_atom[node[node_id].branch_atom_serial].crd.z};
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			node[node_id].n = tempn;
			node[node_id].a = tempa;
		}
		//__syncthreads();//这里实际不需要同步

		// 计算原子力和总能量
		float total_energy_in_thread = 0.f;
		// float lig_fsc_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VINA_ATOM atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VINA_ATOM atom_i = vina_atom[i];
			VECTOR force_i = {0.f, 0.f, 0.f};
			VECTOR dr;
			if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
			{
				// box interaction (transbox)
				dr.x = fdimf(transbox_min.x, atom_i.crd.x); // 如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
				dr.y = fdimf(transbox_min.y, atom_i.crd.y);
				dr.z = fdimf(transbox_min.z, atom_i.crd.z);
				force_i.x += box_border_strenth * dr.x;
				force_i.y += box_border_strenth * dr.y;
				force_i.z += box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				dr.x = fdimf(atom_i.crd.x, transbox_max.x);
				dr.y = fdimf(atom_i.crd.y, transbox_max.y);
				dr.z = fdimf(atom_i.crd.z, transbox_max.z);
				force_i.x -= box_border_strenth * dr.x;
				force_i.y -= box_border_strenth * dr.y;
				force_i.z -= box_border_strenth * dr.z;
				total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				// protein interaction
				VECTOR serial; // 在蛋白插值网格中的格点坐标
				serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
				// ans = { 0.f,0.f,0.f,0.f };
				total_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;
			}
			// ligand intra interations
			inner_list_start = i * atom_numbers;
			int inner_numbers = inner_interaction_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = inner_interaction_list[inner_list_start + k];
				atom_j = vina_atom[j];
				dr = {atom_i.crd.x - atom_j.crd.x, atom_i.crd.y - atom_j.crd.y, atom_i.crd.z - atom_j.crd.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;
					// gauss1
					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					total_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;
					// gauss2
					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					total_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;
					// repulsion
					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					total_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;
					// hydrophobic
					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						total_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}
					// H bond
					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						total_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}

					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&frc[j].x, -temp_force.x);
					atomicAdd(&frc[j].y, -temp_force.y);
					atomicAdd(&frc[j].z, -temp_force.z);
				}
			}
			// Ligand-SideChain interactions;

			for (int j = 0; j < fsc_atom_numbers; j = j + 1)
			{

				atom_j = fsc_vina_atom[j];

				dr = {atom_i.crd.x - atom_j.crd.x,
					  atom_i.crd.y - atom_j.crd.y,
					  atom_i.crd.z - atom_j.crd.z};

				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度

				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;

					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					total_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					total_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					total_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;

					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						total_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}

					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						total_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}

					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					// NO atomicAdd
					// printf("%f\n", total_energy_in_thread);
				}
				// printf("%d %d %.3f \n" ,i,j,rij);
			}
			atomicAdd(&frc[i].x, force_i.x);
			atomicAdd(&frc[i].y, force_i.y);
			atomicAdd(&frc[i].z, force_i.z);
		}
		// printf("lig_fsc_energy_in_thread: %f\n", lig_fsc_energy_in_thread);
		atomicAdd(&shared_data[0], total_energy_in_thread);
		__syncthreads(); // 能量加和完全，且梯度以及node的叉乘相关信息完全

		// 提前退出优化（开起这个竟然变慢很多，因此目前只能固定次数优化，但理论上应足够够用）
		// if (fabsf(shared_data[0] - shared_data[1]) < CONVERGENCE_CUTOFF)
		//{
		//	//opt_i = MAX_OPTIMIZE_STEPS;
		//	if (threadIdx.x == 0)
		//	{
		//		//energy[0] = shared_data[0];
		//	}
		//	//break;
		// }
		if (threadIdx.x == 0)
		{
			energy[0] = shared_data[0];
			shared_data[1] = shared_data[0];
			shared_data[0] = 0.f;
			alpha1[0] = 0.f;
			alpha1[1] = 0.f;
			alpha2[0] = 0.f;
			alpha2[1] = 0.f;
			alpha3[0] = 0.f;
			alpha3[1] = 0.f;
		}

		// 计算广义力
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = {vina_atom[0].crd.x, vina_atom[0].crd.y, vina_atom[0].crd.z};
			VECTOR temp_crd2 = {vina_atom[i].crd.x, vina_atom[i].crd.y, vina_atom[i].crd.z};
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&dU_du_crd[u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&dU_du_crd[u_freedom - 2], (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]));
			atomicAdd(&dU_du_crd[u_freedom - 3], (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12])));

			atomicAdd(&dU_du_crd[u_freedom - 6], temp_frc.x);
			atomicAdd(&dU_du_crd[u_freedom - 5], temp_frc.y);
			atomicAdd(&dU_du_crd[u_freedom - 4], temp_frc.z);

			int current_node_id = atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - node[current_node_id].a.z;
				rot_axis = node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		// 进行BB优化更新(暂时未区分整体转动、平动和二面角自由度的各自优化)
		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float s = u_crd[i] - last_u_crd[i];
			float y = dU_du_crd[i] - last_dU_du_crd[i];
			last_u_crd[i] = u_crd[i];
			last_dU_du_crd[i] = dU_du_crd[i];
			if (i < u_freedom - 6)
			{
				atomicAdd(&alpha1[0], y * s);
				atomicAdd(&alpha1[1], y * y);
			}
			else if (i < u_freedom - 3)
			{
				atomicAdd(&alpha2[0], y * s);
				atomicAdd(&alpha2[1], y * y);
			}
			else
			{
				atomicAdd(&alpha3[0], y * s);
				atomicAdd(&alpha3[1], y * y);
			}
		}
		__syncthreads();

		for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
		{
			float du;
			if (i < u_freedom - 6)
			{
				float temp_alpha = fabsf(alpha1[0]) / fmaxf(alpha1[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			}
			else if (i < u_freedom - 3)
			{
				float temp_alpha = fabsf(alpha2[0]) / fmaxf(alpha2[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 1.f / 10000.f), du);
			}
			else
			{
				float temp_alpha = fabsf(alpha3[0]) / fmaxf(alpha3[1], 1.e-6f);
				du = temp_alpha * dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 2.f * 3.141592654f / 100000.f), du);
			}
			dU_du_crd[i] = 0.f;
			u_crd[i] += du; // temp muted
		}
		__syncthreads();
	}
}

// 对pair作用不使用插值表，直接进行计算
// 4.24 新添加 modeFlex

__global__ void Optimize_Structure_BB2_Direct_Pair_Device_modeSC(
	const int atom_numbers, const int *inner_interaction_list, const float cutoff,
	const int *atom_to_node_serial,
	const VECTOR *ref_crd, VINA_ATOM *vina_atom, VECTOR *frc, float *energy, float *inter_energy,
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR box_min, const VECTOR box_max, const VECTOR protein_mesh_grid_length_inverse,
	const int u_freedom, float *u_crd, float *last_u_crd, float *dU_du_crd, float *last_dU_du_crd,
	const int node_numbers, NODE *node,
	const int lig_atom_numbers, VINA_ATOM *lig_vina_atom)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[23];
	float *rot_matrix = &shared_data[2];
	float *alpha1 = &shared_data[17];

	if (threadIdx.x == 0)
	{
		shared_data[0] = 0.f;		 // 临时能量项
		shared_data[1] = BIG_ENERGY; // 临时能量项
		shared_data[20] = 0.f;		 // 临时inter能量项
		shared_data[21] = BIG_ENERGY;
	}

	// 进入主循环前的基本初始化
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		dU_du_crd[i] = 0.f;
		last_dU_du_crd[i] = 0.f;
		last_u_crd[i] = u_crd[i];
	}

	// 进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		// 在当前广义坐标下更新各转动矩阵
		for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
		{
			if (i != node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				node[i].matrix[0] = temp_matrix_1[0];
				node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				node[i].matrix[4] = temp_matrix_1[4];
				node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
				sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
				sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

				rot_matrix[0] = cos_b * cos_c;
				rot_matrix[1] = cos_b * sin_c;
				rot_matrix[2] = -sin_b;
				rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				rot_matrix[5] = cos_b * sin_a;
				rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				rot_matrix[8] = cos_a * cos_b;

				shared_data[11] = cos_b;
				shared_data[12] = sin_b;
				shared_data[13] = cos_a;
				shared_data[14] = sin_a;
				shared_data[15] = rot_matrix[8]; // cacb
				shared_data[16] = rot_matrix[5]; // cbsa
			}
		}
		__syncthreads();

		// 由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = atom_to_node_serial[i];
			frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ref_crd[i];
			// printf("ref_crd[%d] = %.3f, %.3f, %.3f \n", i ,ref_crd[i].x, ref_crd[i].y, ref_crd[i].z);
			VECTOR temp_crd2;
			const VECTOR center = ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += node[current_node_id].a0.x;
				temp_crd1.y += node[current_node_id].a0.y;
				temp_crd1.z += node[current_node_id].a0.z;

				current_node_id = node[current_node_id].last_node_serial;
			}

			// 删除平移转动：4.24
			// float identity[9] = {1,0,0, 0,1,0, 0,0,1};
			// temp_crd1.x -= center.x;//整体转动的参考原点总是第一个原子（root原子）
			// temp_crd1.y -= center.y;
			// temp_crd1.z -= center.z;
			// Matrix_Multiply_Vector(&temp_crd2, identity, &temp_crd1);
			vina_atom[i].crd.x = temp_crd1.x + u_crd[u_freedom - 6];
			vina_atom[i].crd.y = temp_crd1.y + u_crd[u_freedom - 5];
			vina_atom[i].crd.z = temp_crd1.z + u_crd[u_freedom - 4];
			// printf("atom %d x = %.3f + %.3f + %.3f\n", i, temp_crd2.x, u_crd[u_freedom - 6], center.x);
		}
		__syncthreads();

		// 由当前坐标更新node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
		for (int node_id = threadIdx.x; node_id < node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = {vina_atom[node[node_id].root_atom_serial].crd.x,
					 vina_atom[node[node_id].root_atom_serial].crd.y,
					 vina_atom[node[node_id].root_atom_serial].crd.z};
			tempn = {vina_atom[node[node_id].branch_atom_serial].crd.x,
					 vina_atom[node[node_id].branch_atom_serial].crd.y,
					 vina_atom[node[node_id].branch_atom_serial].crd.z};
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			node[node_id].n = tempn;
			node[node_id].a = tempa;
		}
		//__syncthreads();//这里实际不需要同步

		// 计算原子力和总能量
		float total_energy_in_thread = 0.f;
		float fsc_lig_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VINA_ATOM atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VINA_ATOM atom_i = vina_atom[i];
			VECTOR force_i = {0.f, 0.f, 0.f};
			VECTOR dr;
			if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
			{
				// box interaction
				// FIXME 侧链不需要盒子
				// dr.x = fdimf(box_min.x, atom_i.crd.x); // 如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
				// dr.y = fdimf(box_min.y, atom_i.crd.y);
				// dr.z = fdimf(box_min.z, atom_i.crd.z);
				// force_i.x += box_border_strenth * dr.x;
				// force_i.y += box_border_strenth * dr.y;
				// force_i.z += box_border_strenth * dr.z;
				// total_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
				// printf("atom #%d box interaction: %.3f\n", i,  0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z));
				// dr.x = fdimf(atom_i.crd.x, box_max.x);
				// dr.y = fdimf(atom_i.crd.y, box_max.y);
				// dr.z = fdimf(atom_i.crd.z, box_max.z);
				// force_i.x -= box_border_strenth * dr.x;
				// force_i.y -= box_border_strenth * dr.y;
				// force_i.z -= box_border_strenth * dr.z;
				// total_energy_in_thread += 0 * 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				// printf("before interact, atom #%d crd = %.3f, %.3f, %.3f\n", i, atom_i.crd.x, atom_i.crd.y, atom_i.crd.z);
				// printf("before interact, atom #%d box_min = %.3f, %.3f, %.3f\n", i, box_min.x, box_min.y, box_min.z);
				VECTOR serial; // 在蛋白插值网格中的格点坐标
				serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
				// ans = { 0.f,0.f,0.f,0.f };
				total_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;

				// printf("atom #%d protein interaction: %.3f\n", i, ans.w);
			}
			inner_list_start = i * atom_numbers;
			int inner_numbers = inner_interaction_list[inner_list_start];
			// fsc-fsc interaction
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = inner_interaction_list[inner_list_start + k];
				atom_j = vina_atom[j];
				dr = {atom_i.crd.x - atom_j.crd.x, atom_i.crd.y - atom_j.crd.y, atom_i.crd.z - atom_j.crd.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;

					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					total_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					total_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					total_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;

					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						total_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}

					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						total_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}

					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&frc[j].x, -temp_force.x);
					atomicAdd(&frc[j].y, -temp_force.y);
					atomicAdd(&frc[j].z, -temp_force.z);
				}
			}
			// Ligand-SideChain Interactions

			for (int j = 0; j < lig_atom_numbers; j = j + 1)
			{
				atom_j = lig_vina_atom[j];
				// FIXME dr is incorrect?

				dr = {atom_i.crd.x - atom_j.crd.x,
					  atom_i.crd.y - atom_j.crd.y,
					  atom_i.crd.z - atom_j.crd.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度

				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;

					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					fsc_lig_energy_in_thread += temp_record;
					total_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					fsc_lig_energy_in_thread += temp_record;
					total_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					fsc_lig_energy_in_thread += temp_record * surface_distance;
					total_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;

					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						fsc_lig_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						total_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}

					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						fsc_lig_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						total_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}
					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					// NO atomicAdd
				}

				// printf("sc%d,lig%d = %.3f  ; E = %.3f\n" ,i,j,rij,total_energy_in_thread);
			}

			atomicAdd(&frc[i].x, force_i.x);
			atomicAdd(&frc[i].y, force_i.y);
			atomicAdd(&frc[i].z, force_i.z);
		}
		atomicAdd(&shared_data[0], total_energy_in_thread);
		atomicAdd(&shared_data[20], fsc_lig_energy_in_thread);
		__syncthreads(); // 能量加和完全，且梯度以及node的叉乘相关信息完全

		// 提前退出优化（开起这个竟然变慢很多，因此目前只能固定次数优化，但理论上应足够够用）
		// if (fabsf(shared_data[0] - shared_data[1]) < CONVERGENCE_CUTOFF)
		//{
		//	//opt_i = MAX_OPTIMIZE_STEPS;
		//	if (threadIdx.x == 0)
		//	{
		//		//energy[0] = shared_data[0];
		//	}
		//	//break;
		// }
		if (threadIdx.x == 0)
		{
			energy[0] = shared_data[0];
			shared_data[1] = shared_data[0];
			shared_data[0] = 0.f;
			alpha1[0] = 0.f;
			alpha1[1] = 0.f;
			inter_energy[0] = shared_data[20];
			shared_data[21] = shared_data[20];
			shared_data[20] = 0.f;
		}

		// 计算广义力
		for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = {vina_atom[0].crd.x, vina_atom[0].crd.y, vina_atom[0].crd.z};
			VECTOR temp_crd2 = {vina_atom[i].crd.x, vina_atom[i].crd.y, vina_atom[i].crd.z};
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = frc[i];
			VECTOR cross;
			VECTOR rot_axis;
			/*
			* 以下不操作
			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&dU_du_crd[u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&dU_du_crd[u_freedom - 2], (-temp_frc.x * (temp_crd.y * shared_data[14] + temp_crd.z * shared_data[13]) + temp_frc.y * temp_crd.x * shared_data[14] + temp_frc.z * temp_crd.x * shared_data[13]));
			atomicAdd(&dU_du_crd[u_freedom - 3], (temp_frc.x * (temp_crd.y * shared_data[15] - temp_crd.z * shared_data[16]) - temp_frc.y * (temp_crd.x * shared_data[15] + temp_crd.z * shared_data[12]) + temp_frc.z * (temp_crd.x * shared_data[16] + temp_crd.y * shared_data[12])));

			atomicAdd(&dU_du_crd[u_freedom - 6], temp_frc.x);
			atomicAdd(&dU_du_crd[u_freedom - 5], temp_frc.y);
			atomicAdd(&dU_du_crd[u_freedom - 4], temp_frc.z);
			*/
			int current_node_id = atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - node[current_node_id].a.z;
				rot_axis = node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		// 进行BB优化更新
		// 更新：现在只操作二面角，u_freedom-6
		for (int i = threadIdx.x; i < u_freedom - 6; i = i + blockDim.x)
		{
			float s = u_crd[i] - last_u_crd[i];
			float y = dU_du_crd[i] - last_dU_du_crd[i];
			last_u_crd[i] = u_crd[i];
			last_dU_du_crd[i] = dU_du_crd[i];

			// 只有二面角自由度
			atomicAdd(&alpha1[0], y * s);
			atomicAdd(&alpha1[1], y * y);
		}
		__syncthreads();

		// 更新：u_freedom-6
		for (int i = threadIdx.x; i < u_freedom - 6; i = i + blockDim.x)
		{
			float du;
			// torsion du
			float temp_alpha = fabsf(alpha1[0]) / fmaxf(alpha1[1], 1.e-6f);
			du = temp_alpha * dU_du_crd[i];
			du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			dU_du_crd[i] = 0.f;

			u_crd[i] += du; // no muted;
		}
		__syncthreads();
	}
}
__global__ void Update_Structure_LIG(
	const int atom_numbers, const int *atom_to_node_serial,
	const VECTOR *ref_crd, VINA_ATOM *vina_atom,
	const int u_freedom, float *u_crd, float *last_u_crd,
	const int node_numbers, NODE *node)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[23];
	float *rot_matrix = &shared_data[2];

	// 进入主循环前的基本初始化
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		last_u_crd[i] = u_crd[i];
	}
	// 进入主循环前，先同步
	__syncthreads();

	// 在当前广义坐标下更新各转动矩阵
	for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
	{
		if (i != node_numbers)
		{
			float temp_matrix_1[9];
			float cosa, sina, cosa_1;
			sincosf(u_crd[i], &sina, &cosa);
			cosa_1 = 1.f - cosa;
			VECTOR temp_n0 = node[i].n0;
			temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
			temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
			temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
			temp_matrix_1[3] = temp_matrix_1[1];
			temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
			temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
			temp_matrix_1[6] = temp_matrix_1[2];
			temp_matrix_1[7] = temp_matrix_1[5];
			temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

			node[i].matrix[0] = temp_matrix_1[0];
			node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
			node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
			node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
			node[i].matrix[4] = temp_matrix_1[4];
			node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
			node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
			node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
			node[i].matrix[8] = temp_matrix_1[8];
		}
		else
		{
			float cos_c;
			float sin_c;
			float cos_b;
			float sin_b;
			float cos_a;
			float sin_a;
			sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
			sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
			sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

			rot_matrix[0] = cos_b * cos_c;
			rot_matrix[1] = cos_b * sin_c;
			rot_matrix[2] = -sin_b;
			rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
			rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
			rot_matrix[5] = cos_b * sin_a;
			rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
			rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
			rot_matrix[8] = cos_a * cos_b;

			shared_data[11] = cos_b;
			shared_data[12] = sin_b;
			shared_data[13] = cos_a;
			shared_data[14] = sin_a;
			shared_data[15] = rot_matrix[8]; // cacb
			shared_data[16] = rot_matrix[5]; // cbsa
		}
	}
	__syncthreads();
	// 由各转动矩阵和原始坐标生成当前坐标
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
	{
		int current_node_id = atom_to_node_serial[i];
		// frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
		VECTOR temp_crd1 = ref_crd[i];
		VECTOR temp_crd2;
		const VECTOR center = ref_crd[0];
		while (current_node_id != -1)
		{
			temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
			temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
			temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

			Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

			temp_crd1.x += node[current_node_id].a0.x;
			temp_crd1.y += node[current_node_id].a0.y;
			temp_crd1.z += node[current_node_id].a0.z;

			current_node_id = node[current_node_id].last_node_serial;
		}

		temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
		temp_crd1.y -= center.y;
		temp_crd1.z -= center.z;
		Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
		vina_atom[i].crd.x = temp_crd2.x + u_crd[u_freedom - 6] + center.x; // 整体平移在最后加上
		vina_atom[i].crd.y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
		vina_atom[i].crd.z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
	}
	__syncthreads();
}
//
__global__ void Update_Structure_SC(
	const int atom_numbers, const int *atom_to_node_serial,
	const VECTOR *ref_crd, VINA_ATOM *vina_atom,
	const int u_freedom, float *u_crd, float *last_u_crd,
	const int node_numbers, NODE *node)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[23];
	float *rot_matrix = &shared_data[2];

	// 进入主循环前的基本初始化
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		last_u_crd[i] = u_crd[i];
	}

	// 进入主循环前，先同步
	__syncthreads();

	// 在当前广义坐标下更新各转动矩阵
	for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
	{
		if (i != node_numbers)
		{
			float temp_matrix_1[9];
			float cosa, sina, cosa_1;
			sincosf(u_crd[i], &sina, &cosa);
			cosa_1 = 1.f - cosa;
			VECTOR temp_n0 = node[i].n0;
			temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
			temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
			temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
			temp_matrix_1[3] = temp_matrix_1[1];
			temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
			temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
			temp_matrix_1[6] = temp_matrix_1[2];
			temp_matrix_1[7] = temp_matrix_1[5];
			temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

			node[i].matrix[0] = temp_matrix_1[0];
			node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
			node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
			node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
			node[i].matrix[4] = temp_matrix_1[4];
			node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
			node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
			node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
			node[i].matrix[8] = temp_matrix_1[8];
		}
		else
		{
			float cos_c;
			float sin_c;
			float cos_b;
			float sin_b;
			float cos_a;
			float sin_a;
			sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
			sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
			sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

			rot_matrix[0] = cos_b * cos_c;
			rot_matrix[1] = cos_b * sin_c;
			rot_matrix[2] = -sin_b;
			rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
			rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
			rot_matrix[5] = cos_b * sin_a;
			rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
			rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
			rot_matrix[8] = cos_a * cos_b;

			shared_data[11] = cos_b;
			shared_data[12] = sin_b;
			shared_data[13] = cos_a;
			shared_data[14] = sin_a;
			shared_data[15] = rot_matrix[8]; // cacb
			shared_data[16] = rot_matrix[5]; // cbsa
		}
	}
	__syncthreads();

	// 由各转动矩阵和原始坐标生成当前坐标
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
	{
		int current_node_id = atom_to_node_serial[i];
		// frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
		VECTOR temp_crd1 = ref_crd[i];
		// printf("ref_crd[%d] = %.3f, %.3f, %.3f \n", i ,ref_crd[i].x, ref_crd[i].y, ref_crd[i].z);
		VECTOR temp_crd2;
		// const VECTOR center = ref_crd[0];
		while (current_node_id != -1)
		{
			temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
			temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
			temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

			Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

			temp_crd1.x += node[current_node_id].a0.x;
			temp_crd1.y += node[current_node_id].a0.y;
			temp_crd1.z += node[current_node_id].a0.z;

			current_node_id = node[current_node_id].last_node_serial;
		}

		// 删除平移转动：4.24
		// float identity[9] = {1,0,0, 0,1,0, 0,0,1};
		// temp_crd1.x -= center.x;//整体转动的参考原点总是第一个原子（root原子）
		// temp_crd1.y -= center.y;
		// temp_crd1.z -= center.z;
		// Matrix_Multiply_Vector(&temp_crd2, identity, &temp_crd1);
		vina_atom[i].crd.x = temp_crd1.x + u_crd[u_freedom - 6];
		vina_atom[i].crd.y = temp_crd1.y + u_crd[u_freedom - 5];
		vina_atom[i].crd.z = temp_crd1.z + u_crd[u_freedom - 4];
		// printf("atom %d x = %.3f + %.3f + %.3f\n", i, temp_crd2.x, u_crd[u_freedom - 6], center.x);
	}
	__syncthreads();
}

__global__ void Optimize_All_Structure_BB2_Direct_Pair_Device(
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR protein_mesh_grid_length_inverse,
	const VECTOR box_min, const VECTOR box_max,
	const VECTOR transbox_min, const VECTOR transbox_max,
	const float cutoff,
	PARTIAL_RIGID_SMALL_MOLECULE::GPU ligand,
	FLEXIBLE_SIDE_CHAIN::GPU flex_chains,
	float *l_energy, float *f_energy, float *inter_energy,
	const float flex_ratio)
{
	const int &l_atom_numbers = ligand.atom_numbers;
	int *&l_inner_interaction_list = ligand.inner_neighbor_list;
	const int &l_node_numbers = ligand.node_numbers;
	NODE *&l_node = ligand.node;
	int *&l_atom_to_node_serial = ligand.atom_to_node_serial;
	VECTOR *&l_ref_crd = ligand.ref_crd;
	VINA_ATOM *&l_vina_atom = ligand.d_vina_atom;
	VECTOR *&l_frc = ligand.frc;
	const int &l_u_freedom = ligand.u_freedom;
	float *&l_u_crd = ligand.u_crd;
	float *&l_last_u_crd = ligand.last_u_crd;
	float *&l_dU_du_crd = ligand.dU_du_crd;
	float *&l_last_dU_du_crd = ligand.last_dU_du_crd;

	const int &f_atom_numbers = flex_chains.atom_numbers;
	int *&f_inner_interaction_list = flex_chains.inner_neighbor_list;
	const int &f_node_numbers = flex_chains.node_numbers;
	NODE *&f_node = flex_chains.node;
	int *&f_atom_to_node_serial = flex_chains.atom_to_node_serial;
	VECTOR *&f_ref_crd = flex_chains.ref_crd;
	VINA_ATOM *&f_vina_atom = flex_chains.d_vina_atom;
	VECTOR *&f_frc = flex_chains.frc;
	const int &f_u_freedom = flex_chains.u_freedom;
	float *&f_u_crd = flex_chains.u_crd;
	float *&f_last_u_crd = flex_chains.last_u_crd;
	float *&f_dU_du_crd = flex_chains.dU_du_crd;
	float *&f_last_dU_du_crd = flex_chains.last_dU_du_crd;
	// look like optimization happened

	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float l_shared_data[23];
	float *l_rot_matrix = &l_shared_data[2];
	float *l_alpha1 = &l_shared_data[17];
	float *l_alpha2 = &l_shared_data[19];
	float *l_alpha3 = &l_shared_data[21];
	if (threadIdx.x == 0)
	{
		l_shared_data[0] = 0.f;		   // 临时能量项
		l_shared_data[1] = BIG_ENERGY; // 临时能量项
	}
	__shared__ float f_shared_data[23];
	float *f_rot_matrix = &f_shared_data[2];
	float *f_alpha1 = &f_shared_data[17];
	if (threadIdx.x == 0)
	{
		f_shared_data[0] = 0.f;		   // 临时能量项
		f_shared_data[1] = BIG_ENERGY; // 临时能量项
		f_shared_data[20] = 0.f;	   // 临时inter能量项
		f_shared_data[21] = BIG_ENERGY;
	}
	// 进入主循环前的基本初始化1
	for (int i = threadIdx.x; i < l_u_freedom; i = i + blockDim.x)
	{
		l_dU_du_crd[i] = 0.f;
		l_last_dU_du_crd[i] = 0.f;
		l_last_u_crd[i] = l_u_crd[i];
	}

	// 进入主循环前的基本初始化2
	for (int i = threadIdx.x; i < f_u_freedom; i = i + blockDim.x)
	{
		f_dU_du_crd[i] = 0.f;
		f_last_dU_du_crd[i] = 0.f;
		f_last_u_crd[i] = f_u_crd[i];
	}
	// 进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIMIZE_STEPS; opt_i += 1)
	{
		// 在当前广义坐标下更新ligand的各转动矩阵
		for (int i = threadIdx.x; i <= l_node_numbers; i = i + blockDim.x)
		{
			if (i != l_node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(l_u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = l_node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				l_node[i].matrix[0] = temp_matrix_1[0];
				l_node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				l_node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				l_node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				l_node[i].matrix[4] = temp_matrix_1[4];
				l_node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				l_node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				l_node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				l_node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(l_u_crd[l_u_freedom - 3], &sin_c, &cos_c);
				sincosf(l_u_crd[l_u_freedom - 2], &sin_b, &cos_b);
				sincosf(l_u_crd[l_u_freedom - 1], &sin_a, &cos_a);

				l_rot_matrix[0] = cos_b * cos_c;
				l_rot_matrix[1] = cos_b * sin_c;
				l_rot_matrix[2] = -sin_b;
				l_rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				l_rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				l_rot_matrix[5] = cos_b * sin_a;
				l_rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				l_rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				l_rot_matrix[8] = cos_a * cos_b;

				l_shared_data[11] = cos_b;
				l_shared_data[12] = sin_b;
				l_shared_data[13] = cos_a;
				l_shared_data[14] = sin_a;
				l_shared_data[15] = l_rot_matrix[8]; // cacb
				l_shared_data[16] = l_rot_matrix[5]; // cbsa
			}
		}
		__syncthreads();
		// 在当前广义坐标下更新fsc各转动矩阵
		for (int i = threadIdx.x; i <= f_node_numbers; i = i + blockDim.x)
		{
			if (i != f_node_numbers)
			{
				float temp_matrix_1[9];
				float cosa, sina, cosa_1;
				sincosf(f_u_crd[i], &sina, &cosa);
				cosa_1 = 1.f - cosa;
				VECTOR temp_n0 = f_node[i].n0;
				temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
				temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
				temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
				temp_matrix_1[3] = temp_matrix_1[1];
				temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
				temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
				temp_matrix_1[6] = temp_matrix_1[2];
				temp_matrix_1[7] = temp_matrix_1[5];
				temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

				f_node[i].matrix[0] = temp_matrix_1[0];
				f_node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
				f_node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
				f_node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
				f_node[i].matrix[4] = temp_matrix_1[4];
				f_node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
				f_node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
				f_node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
				f_node[i].matrix[8] = temp_matrix_1[8];
			}
			else
			{
				float cos_c;
				float sin_c;
				float cos_b;
				float sin_b;
				float cos_a;
				float sin_a;
				sincosf(f_u_crd[f_u_freedom - 3], &sin_c, &cos_c);
				sincosf(f_u_crd[f_u_freedom - 2], &sin_b, &cos_b);
				sincosf(f_u_crd[f_u_freedom - 1], &sin_a, &cos_a);

				f_rot_matrix[0] = cos_b * cos_c;
				f_rot_matrix[1] = cos_b * sin_c;
				f_rot_matrix[2] = -sin_b;
				f_rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
				f_rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
				f_rot_matrix[5] = cos_b * sin_a;
				f_rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
				f_rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
				f_rot_matrix[8] = cos_a * cos_b;

				f_shared_data[11] = cos_b;
				f_shared_data[12] = sin_b;
				f_shared_data[13] = cos_a;
				f_shared_data[14] = sin_a;
				f_shared_data[15] = f_rot_matrix[8]; // cacb
				f_shared_data[16] = f_rot_matrix[5]; // cbsa
			}
		}
		__syncthreads();

		// 由各转动矩阵和原始坐标生成当前lig坐标
		for (int i = threadIdx.x; i < l_atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = l_atom_to_node_serial[i];
			l_frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = l_ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = l_ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - l_node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - l_node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - l_node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, l_node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += l_node[current_node_id].a0.x;
				temp_crd1.y += l_node[current_node_id].a0.y;
				temp_crd1.z += l_node[current_node_id].a0.z;

				current_node_id = l_node[current_node_id].last_node_serial;
			}

			temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, l_rot_matrix, &temp_crd1);
			l_vina_atom[i].crd.x = temp_crd2.x + l_u_crd[l_u_freedom - 6] + center.x; // 整体平移在最后加上
			l_vina_atom[i].crd.y = temp_crd2.y + l_u_crd[l_u_freedom - 5] + center.y;
			l_vina_atom[i].crd.z = temp_crd2.z + l_u_crd[l_u_freedom - 4] + center.z;
		}
		__syncthreads();
		// 由各转动矩阵和原始坐标生成当前fsc坐标
		for (int i = threadIdx.x; i < f_atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = f_atom_to_node_serial[i];
			f_frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = f_ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = f_ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - f_node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - f_node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - f_node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, f_node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += f_node[current_node_id].a0.x;
				temp_crd1.y += f_node[current_node_id].a0.y;
				temp_crd1.z += f_node[current_node_id].a0.z;

				current_node_id = f_node[current_node_id].last_node_serial;
			}

			// 删除平移转动：4.24
			// float identity[9] = {1,0,0, 0,1,0, 0,0,1};
			// temp_crd1.x -= center.x;//整体转动的参考原点总是第一个原子（root原子）
			// temp_crd1.y -= center.y;
			// temp_crd1.z -= center.z;
			// Matrix_Multiply_Vector(&temp_crd2, identity, &temp_crd1);
			f_vina_atom[i].crd.x = temp_crd1.x + f_u_crd[f_u_freedom - 6];
			f_vina_atom[i].crd.y = temp_crd1.y + f_u_crd[f_u_freedom - 5];
			f_vina_atom[i].crd.z = temp_crd1.z + f_u_crd[f_u_freedom - 4];
			// printf("atom %d x = %.3f + %.3f + %.3f\n", i, temp_crd2.x, u_crd[u_freedom - 6], center.x);
		}
		__syncthreads();
		// 由当前坐标更新l_node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
		for (int node_id = threadIdx.x; node_id < l_node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = {l_vina_atom[l_node[node_id].root_atom_serial].crd.x,
					 l_vina_atom[l_node[node_id].root_atom_serial].crd.y,
					 l_vina_atom[l_node[node_id].root_atom_serial].crd.z};
			tempn = {l_vina_atom[l_node[node_id].branch_atom_serial].crd.x,
					 l_vina_atom[l_node[node_id].branch_atom_serial].crd.y,
					 l_vina_atom[l_node[node_id].branch_atom_serial].crd.z};
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			l_node[node_id].n = tempn;
			l_node[node_id].a = tempa;
		}
		__syncthreads(); // 这里实际不需要同步
		// FIXME 不知道现在是否需要同步

		// 由当前坐标更新f_node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
		for (int node_id = threadIdx.x; node_id < f_node_numbers; node_id = node_id + blockDim.x)
		{
			float temp_length;
			VECTOR tempa, tempn;
			tempa = {f_vina_atom[f_node[node_id].root_atom_serial].crd.x,
					 f_vina_atom[f_node[node_id].root_atom_serial].crd.y,
					 f_vina_atom[f_node[node_id].root_atom_serial].crd.z};
			tempn = {f_vina_atom[f_node[node_id].branch_atom_serial].crd.x,
					 f_vina_atom[f_node[node_id].branch_atom_serial].crd.y,
					 f_vina_atom[f_node[node_id].branch_atom_serial].crd.z};
			tempn.x -= tempa.x;
			tempn.y -= tempa.y;
			tempn.z -= tempa.z;
			temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
			tempn.x *= temp_length;
			tempn.y *= temp_length;
			tempn.z *= temp_length;
			f_node[node_id].n = tempn;
			f_node[node_id].a = tempa;
		}
		__syncthreads(); // 这里实际不需要同步
						 // FIXME 不知道现在是否需要同步

		// 计算原子力和总能量 Ligand
		// 在这个版本，能量同时计算
		float total_energy_in_thread = 0.f;
		float lig_energy_in_thread = 0.f;

		// float lig_fsc_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < l_atom_numbers; i = i + blockDim.x)
		{
			VINA_ATOM atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VINA_ATOM atom_i = l_vina_atom[i];
			VECTOR force_i = {0.f, 0.f, 0.f};
			VECTOR dr;
			// 1. Ligand - receptor(+box) interaction
			if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
			{
				// box interaction (transbox)
				dr.x = fdimf(transbox_min.x, atom_i.crd.x); // 如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
				dr.y = fdimf(transbox_min.y, atom_i.crd.y);
				dr.z = fdimf(transbox_min.z, atom_i.crd.z);
				force_i.x += box_border_strenth * dr.x;
				force_i.y += box_border_strenth * dr.y;
				force_i.z += box_border_strenth * dr.z;
				lig_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				dr.x = fdimf(atom_i.crd.x, transbox_max.x);
				dr.y = fdimf(atom_i.crd.y, transbox_max.y);
				dr.z = fdimf(atom_i.crd.z, transbox_max.z);
				force_i.x -= box_border_strenth * dr.x;
				force_i.y -= box_border_strenth * dr.y;
				force_i.z -= box_border_strenth * dr.z;
				lig_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

				// protein interaction
				VECTOR serial; // 在蛋白插值网格中的格点坐标
				serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
				// ans = { 0.f,0.f,0.f,0.f };
				lig_energy_in_thread += ans.w;
				force_i.x += ans.x;
				force_i.y += ans.y;
				force_i.z += ans.z;
			}
			// 2. ligand intra interations
			inner_list_start = i * l_atom_numbers;
			int inner_numbers = l_inner_interaction_list[inner_list_start];
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = l_inner_interaction_list[inner_list_start + k];
				atom_j = l_vina_atom[j];
				dr = {atom_i.crd.x - atom_j.crd.x, atom_i.crd.y - atom_j.crd.y, atom_i.crd.z - atom_j.crd.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;
					// gauss1
					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					lig_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;
					// gauss2
					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					lig_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;
					// repulsion
					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					lig_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;
					// hydrophobic
					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						lig_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}
					// H bond
					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						lig_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}

					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&l_frc[j].x, -temp_force.x);
					atomicAdd(&l_frc[j].y, -temp_force.y);
					atomicAdd(&l_frc[j].z, -temp_force.z);
				}
			}
			// 3. Ligand-SideChain interactions;
			for (int j = 0; j < f_atom_numbers; j = j + 1)
			{

				atom_j = f_vina_atom[j];

				dr = {atom_i.crd.x - atom_j.crd.x,
					  atom_i.crd.y - atom_j.crd.y,
					  atom_i.crd.z - atom_j.crd.z};

				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度

				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;

					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					lig_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					lig_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					lig_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;

					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						lig_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}

					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						lig_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}

					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					// NO atomicAdd
					// printf("%f\n", lig_energy_in_thread);
				}
				// printf("%d %d %.3f \n" ,i,j,rij);
			}
			atomicAdd(&l_frc[i].x, force_i.x);
			atomicAdd(&l_frc[i].y, force_i.y);
			atomicAdd(&l_frc[i].z, force_i.z);
		}
		// printf("lig_fsc_energy_in_thread: %f\n", lig_fsc_energy_in_thread);
		atomicAdd(&l_shared_data[0], lig_energy_in_thread);
		__syncthreads(); // 能量加和完全，且梯度以及node的叉乘相关信息完全

		float fsc_energy_in_thread = 0.f;
		float fsc_lig_energy_in_thread = 0.f;
		for (int i = threadIdx.x; i < f_atom_numbers; i = i + blockDim.x)
		{
			VINA_ATOM atom_j;
			VECTOR temp_force;
			float rij, dd, dd_, frc_abs, rij_inverse;
			float4 ans;
			int inner_list_start;
			VINA_ATOM atom_i = f_vina_atom[i];
			VECTOR force_i = {0.f, 0.f, 0.f};
			VECTOR dr;

			// 4. sidechain-receptor interaction
			if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
			{
				// no box interaction

				VECTOR serial; // 在蛋白插值网格中的格点坐标
				serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
				serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
				serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
				ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
				// ans = { 0.f,0.f,0.f,0.f };
				fsc_energy_in_thread += ans.w * flex_ratio;
				force_i.x += ans.x * flex_ratio;
				force_i.y += ans.y * flex_ratio;
				force_i.z += ans.z * flex_ratio;

				// printf("atom #%d %d: %f %f %f  protein interaction: %.3f\n", i, atom_i.atom_type, atom_i.crd.x, atom_i.crd.y, atom_i.crd.z, ans.w);
			}
			inner_list_start = i * f_atom_numbers;
			int inner_numbers = f_inner_interaction_list[inner_list_start];
			// 5. fsc-fsc interaction
			for (int k = 1; k <= inner_numbers; k = k + 1)
			{
				int j = f_inner_interaction_list[inner_list_start + k];
				atom_j = f_vina_atom[j];

				dr = {atom_i.crd.x - atom_j.crd.x, atom_i.crd.y - atom_j.crd.y, atom_i.crd.z - atom_j.crd.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;

					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					fsc_energy_in_thread += flex_ratio * temp_record;
					frc_abs = flex_ratio * 2.f * k_gauss1_2 * temp_record * surface_distance;

					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					fsc_energy_in_thread += flex_ratio * temp_record;
					frc_abs += flex_ratio * 2.f * k_gauss2_2 * temp_record * dp;

					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					fsc_energy_in_thread += flex_ratio * temp_record * surface_distance;
					frc_abs += flex_ratio * (-2.f) * temp_record;

					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						fsc_energy_in_thread += flex_ratio * temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += flex_ratio * (-temp_record) * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}

					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						fsc_energy_in_thread += flex_ratio * temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += flex_ratio * (-temp_record) * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}

					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					atomicAdd(&f_frc[j].x, -temp_force.x);
					atomicAdd(&f_frc[j].y, -temp_force.y);
					atomicAdd(&f_frc[j].z, -temp_force.z);
				}
			}
			// 3.' Ligand-SideChain Interactions

			for (int j = 0; j < l_atom_numbers; j = j + 1)
			{
				atom_j = l_vina_atom[j];

				dr = {atom_i.crd.x - atom_j.crd.x,
					  atom_i.crd.y - atom_j.crd.y,
					  atom_i.crd.z - atom_j.crd.z};
				rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度

				if (rij < cutoff)
				{
					float surface_distance = rij - atom_i.radius - atom_j.radius;
					float temp_record;

					temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
					fsc_lig_energy_in_thread += temp_record;
					// fsc_energy_in_thread += temp_record;
					frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

					float dp = surface_distance - k_gauss2_c;
					temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
					fsc_lig_energy_in_thread += temp_record;
					// fsc_energy_in_thread += temp_record;
					frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

					temp_record = k_repulsion * surface_distance * signbit(surface_distance);
					fsc_lig_energy_in_thread += temp_record * surface_distance;
					// fsc_energy_in_thread += temp_record * surface_distance;
					frc_abs += -2.f * temp_record;

					if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
					{
						temp_record = 1.f * k_hydrophobic;
						fsc_lig_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						// fsc_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
						frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
					}

					if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
					{
						temp_record = 1.f * k_h_bond;
						fsc_lig_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						// fsc_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
						frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
					}
					rij_inverse = 1.f / (rij + 10.e-6f);
					frc_abs *= rij_inverse;
					temp_force.x = frc_abs * dr.x;
					temp_force.y = frc_abs * dr.y;
					temp_force.z = frc_abs * dr.z;
					force_i.x += temp_force.x;
					force_i.y += temp_force.y;
					force_i.z += temp_force.z;
					// NO atomicAdd
				}

				// printf("sc%d,lig%d = %.3f  ; E = %.3f\n" ,i,j,rij,fsc_energy_in_thread);
			}

			atomicAdd(&f_frc[i].x, force_i.x);
			atomicAdd(&f_frc[i].y, force_i.y);
			atomicAdd(&f_frc[i].z, force_i.z);
		}
		atomicAdd(&f_shared_data[0], fsc_energy_in_thread); // fsc.energy = fsc-rec + fsc-fsc (intra prot)
		// atomicAdd(&l_shared_data[0], fsc_energy_in_thread);		 // lig.energy = total_energy
		atomicAdd(&f_shared_data[20], fsc_lig_energy_in_thread); // fsc.inter_energy = fsc-lig (inter)
		__syncthreads();										 // 能量加和完全，且梯度以及node的叉乘相关信息完全

		// 提前退出优化（开起这个竟然变慢很多，因此目前只能固定次数优化，但理论上应足够够用）

		// 存储Ligand&SideChain能量
		if (threadIdx.x == 0)
		{
			l_energy[0] = l_shared_data[0]; // ligand energy
			l_shared_data[1] = l_shared_data[0];
			l_shared_data[0] = 0.f;
			l_alpha1[0] = 0.f;
			l_alpha1[1] = 0.f;
			l_alpha2[0] = 0.f;
			l_alpha2[1] = 0.f;
			l_alpha3[0] = 0.f;
			l_alpha3[1] = 0.f;

			f_energy[0] = f_shared_data[0]; // inter ligand-fsc energy
			f_shared_data[1] = f_shared_data[0];
			f_shared_data[0] = 0.f;
			f_alpha1[0] = 0.f;
			f_alpha1[1] = 0.f;
			inter_energy[0] = f_shared_data[20];
			f_shared_data[21] = f_shared_data[20];
			f_shared_data[20] = 0.f;
		}
		// 提前退出优化（开起这个竟然变慢很多，因此目前只能固定次数优化，但理论上应足够够用）

		// 计算ligand广义力
		for (int i = threadIdx.x; i < l_atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = {l_vina_atom[0].crd.x, l_vina_atom[0].crd.y, l_vina_atom[0].crd.z};
			VECTOR temp_crd2 = {l_vina_atom[i].crd.x, l_vina_atom[i].crd.y, l_vina_atom[i].crd.z};
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = l_frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			temp_crd.x = temp_crd2.x - center.x;
			temp_crd.y = temp_crd2.y - center.y;
			temp_crd.z = temp_crd2.z - center.z;

			atomicAdd(&l_dU_du_crd[l_u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
			atomicAdd(&l_dU_du_crd[l_u_freedom - 2], (-temp_frc.x * (temp_crd.y * l_shared_data[14] + temp_crd.z * l_shared_data[13]) + temp_frc.y * temp_crd.x * l_shared_data[14] + temp_frc.z * temp_crd.x * l_shared_data[13]));
			atomicAdd(&l_dU_du_crd[l_u_freedom - 3], (temp_frc.x * (temp_crd.y * l_shared_data[15] - temp_crd.z * l_shared_data[16]) - temp_frc.y * (temp_crd.x * l_shared_data[15] + temp_crd.z * l_shared_data[12]) + temp_frc.z * (temp_crd.x * l_shared_data[16] + temp_crd.y * l_shared_data[12])));

			atomicAdd(&l_dU_du_crd[l_u_freedom - 6], temp_frc.x);
			atomicAdd(&l_dU_du_crd[l_u_freedom - 5], temp_frc.y);
			atomicAdd(&l_dU_du_crd[l_u_freedom - 4], temp_frc.z);

			int current_node_id = l_atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - l_node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - l_node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - l_node[current_node_id].a.z;
				rot_axis = l_node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&l_dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = l_node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();
		// 计算SideChain广义力
		for (int i = threadIdx.x; i < f_atom_numbers; i = i + blockDim.x)
		{
			VECTOR center = {f_vina_atom[0].crd.x, f_vina_atom[0].crd.y, f_vina_atom[0].crd.z};
			VECTOR temp_crd2 = {f_vina_atom[i].crd.x, f_vina_atom[i].crd.y, f_vina_atom[i].crd.z};
			VECTOR temp_crd = temp_crd2;
			VECTOR temp_frc = f_frc[i];
			VECTOR cross;
			VECTOR rot_axis;

			int current_node_id = f_atom_to_node_serial[i];
			while (current_node_id != -1)
			{
				temp_crd.x = temp_crd2.x - f_node[current_node_id].a.x;
				temp_crd.y = temp_crd2.y - f_node[current_node_id].a.y;
				temp_crd.z = temp_crd2.z - f_node[current_node_id].a.z;
				rot_axis = f_node[current_node_id].n;

				cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
				cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
				cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

				atomicAdd(&f_dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
				current_node_id = f_node[current_node_id].last_node_serial;
			}
		}
		__syncthreads();

		// 进行BB优化更新ligand(暂时未区分整体转动、平动和二面角自由度的各自优化)
		for (int i = threadIdx.x; i < l_u_freedom; i = i + blockDim.x)
		{
			float s = l_u_crd[i] - l_last_u_crd[i];
			float y = l_dU_du_crd[i] - l_last_dU_du_crd[i];
			l_last_u_crd[i] = l_u_crd[i];
			l_last_dU_du_crd[i] = l_dU_du_crd[i];
			if (i < l_u_freedom - 6)
			{
				atomicAdd(&l_alpha1[0], y * s);
				atomicAdd(&l_alpha1[1], y * y);
			}
			else if (i < l_u_freedom - 3)
			{
				atomicAdd(&l_alpha2[0], y * s);
				atomicAdd(&l_alpha2[1], y * y);
			}
			else
			{
				atomicAdd(&l_alpha3[0], y * s);
				atomicAdd(&l_alpha3[1], y * y);
			}
		}
		__syncthreads();

		for (int i = threadIdx.x; i < l_u_freedom; i = i + blockDim.x)
		{
			float du;
			if (i < l_u_freedom - 6)
			{
				float temp_alpha = fabsf(l_alpha1[0]) / fmaxf(l_alpha1[1], 1.e-6f);
				du = temp_alpha * l_dU_du_crd[i];
				du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			}
			else if (i < l_u_freedom - 3)
			{
				float temp_alpha = fabsf(l_alpha2[0]) / fmaxf(l_alpha2[1], 1.e-6f);
				du = temp_alpha * l_dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 1.f / 10000.f), du);
			}
			else
			{
				float temp_alpha = fabsf(l_alpha3[0]) / fmaxf(l_alpha3[1], 1.e-6f);
				du = temp_alpha * l_dU_du_crd[i];
				du = copysignf(fmaxf(fabsf(du), 2.f * 3.141592654f / 100000.f), du);
			}
			l_dU_du_crd[i] = 0.f;
			l_u_crd[i] += du; // temp muted
		}
		__syncthreads();
		// 进行BB优化更新side chains
		// 更新：现在只操作二面角，u_freedom-6
		for (int i = threadIdx.x; i < f_u_freedom - 6; i = i + blockDim.x)
		{
			float s = f_u_crd[i] - f_last_u_crd[i];
			float y = f_dU_du_crd[i] - f_last_dU_du_crd[i];
			f_last_u_crd[i] = f_u_crd[i];
			f_last_dU_du_crd[i] = f_dU_du_crd[i];

			// 只有二面角自由度
			atomicAdd(&f_alpha1[0], y * s);
			atomicAdd(&f_alpha1[1], y * y);
		}
		__syncthreads();

		// 更新：u_freedom-6
		for (int i = threadIdx.x; i < f_u_freedom - 6; i = i + blockDim.x)
		{
			float du;
			// torsion du
			float temp_alpha = fabsf(f_alpha1[0]) / fmaxf(f_alpha1[1], 1.e-6f);
			du = temp_alpha * f_dU_du_crd[i];
			du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
			f_dU_du_crd[i] = 0.f;

			f_u_crd[i] += du; // no muted;
		}
		__syncthreads();
	}
}
__global__ void New_Bootstrap_Optimization_Kernel(
	const long long int *protein_mesh, const float box_border_strenth,
	const VECTOR protein_mesh_grid_length_inverse,
	const VECTOR box_min, const VECTOR box_max,
	const VECTOR transbox_min, const VECTOR transbox_max,
	const float cutoff,
	PARTIAL_RIGID_SMALL_MOLECULE::GPU ligand,
	FLEXIBLE_SIDE_CHAIN::GPU flex_chains,
	float *l_energy, float *f_energy, float *inter_energy,
	const float flex_ratio)
{
	const int &l_atom_numbers = ligand.atom_numbers;
	int *&l_inner_interaction_list = ligand.inner_neighbor_list;
	const int &l_node_numbers = ligand.node_numbers;
	NODE *&l_node = ligand.node;
	int *&l_atom_to_node_serial = ligand.atom_to_node_serial;
	VECTOR *&l_ref_crd = ligand.ref_crd; // 需要用ref_crd
	VINA_ATOM *&l_vina_atom = ligand.d_vina_atom;
	VECTOR *&l_frc = ligand.frc;
	const int &l_u_freedom = ligand.u_freedom;
	float *&l_u_crd = ligand.u_crd;
	float *&l_last_u_crd = ligand.last_u_crd;
	float *&l_dU_du_crd = ligand.dU_du_crd;
	float *&l_last_dU_du_crd = ligand.last_dU_du_crd;

	const int &f_atom_numbers = flex_chains.atom_numbers;
	int *&f_inner_interaction_list = flex_chains.inner_neighbor_list;
	const int &f_node_numbers = flex_chains.node_numbers;
	NODE *&f_node = flex_chains.node;
	int *&f_atom_to_node_serial = flex_chains.atom_to_node_serial;
	VECTOR *&f_ref_crd = flex_chains.ref_crd;
	VINA_ATOM *&f_vina_atom = flex_chains.d_vina_atom;
	VECTOR *&f_frc = flex_chains.frc;
	const int &f_u_freedom = flex_chains.u_freedom;
	float *&f_u_crd = flex_chains.u_crd;
	float *&f_last_u_crd = flex_chains.last_u_crd;
	float *&f_dU_du_crd = flex_chains.dU_du_crd;
	float *&f_last_dU_du_crd = flex_chains.last_dU_du_crd;

	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float l_shared_data[23];
	float *l_rot_matrix = &l_shared_data[2];
	float *l_alpha1 = &l_shared_data[17];
	float *l_alpha2 = &l_shared_data[19];
	float *l_alpha3 = &l_shared_data[21];
	if (threadIdx.x == 0)
	{
		l_shared_data[0] = 0.f;		   // 临时能量项
		l_shared_data[1] = BIG_ENERGY; // 临时能量项
	}
	__shared__ float f_shared_data[23];
	float *f_rot_matrix = &f_shared_data[2];
	float *f_alpha1 = &f_shared_data[17];
	if (threadIdx.x == 0)
	{
		f_shared_data[0] = 0.f;		   // 临时能量项
		f_shared_data[1] = BIG_ENERGY; // 临时能量项
		f_shared_data[20] = 0.f;	   // 临时inter能量项
		f_shared_data[21] = BIG_ENERGY;
	}
	// 进入主循环前的基本初始化1
	for (int i = threadIdx.x; i < l_u_freedom; i = i + blockDim.x)
	{
		l_dU_du_crd[i] = 0.f;
		l_last_dU_du_crd[i] = 0.f;
		l_last_u_crd[i] = l_u_crd[i];
	}
	// FIXME __syncthreads();
	// 进入主循环前的基本初始化2
	for (int i = threadIdx.x; i < f_u_freedom; i = i + blockDim.x)
	{
		f_dU_du_crd[i] = 0.f;
		f_last_dU_du_crd[i] = 0.f;
		f_last_u_crd[i] = f_u_crd[i];
	}
	// 进入主循环前，先同步
	__syncthreads();
	for (int opt_i = 0; opt_i < MAX_OPTIM_TURNS; opt_i += 1)
	{
		// FIXME __syncthreads();
		for (int lig_opt_i = 0; lig_opt_i < MAX_OPTIM_STEP_PER_TURN; lig_opt_i++)
		{
			// 在当前广义坐标下更新ligand的各转动矩阵
			for (int i = threadIdx.x; i <= l_node_numbers; i = i + blockDim.x)
			{
				if (i != l_node_numbers)
				{
					float temp_matrix_1[9];
					float cosa, sina, cosa_1;
					sincosf(l_u_crd[i], &sina, &cosa);
					cosa_1 = 1.f - cosa;
					VECTOR temp_n0 = l_node[i].n0;
					temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
					temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
					temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
					temp_matrix_1[3] = temp_matrix_1[1];
					temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
					temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
					temp_matrix_1[6] = temp_matrix_1[2];
					temp_matrix_1[7] = temp_matrix_1[5];
					temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

					l_node[i].matrix[0] = temp_matrix_1[0];
					l_node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
					l_node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
					l_node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
					l_node[i].matrix[4] = temp_matrix_1[4];
					l_node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
					l_node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
					l_node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
					l_node[i].matrix[8] = temp_matrix_1[8];
				}
				else
				{
					float cos_c;
					float sin_c;
					float cos_b;
					float sin_b;
					float cos_a;
					float sin_a;
					sincosf(l_u_crd[l_u_freedom - 3], &sin_c, &cos_c);
					sincosf(l_u_crd[l_u_freedom - 2], &sin_b, &cos_b);
					sincosf(l_u_crd[l_u_freedom - 1], &sin_a, &cos_a);

					l_rot_matrix[0] = cos_b * cos_c;
					l_rot_matrix[1] = cos_b * sin_c;
					l_rot_matrix[2] = -sin_b;
					l_rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
					l_rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
					l_rot_matrix[5] = cos_b * sin_a;
					l_rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
					l_rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
					l_rot_matrix[8] = cos_a * cos_b;

					l_shared_data[11] = cos_b;
					l_shared_data[12] = sin_b;
					l_shared_data[13] = cos_a;
					l_shared_data[14] = sin_a;
					l_shared_data[15] = l_rot_matrix[8]; // cacb
					l_shared_data[16] = l_rot_matrix[5]; // cbsa
				}
			}
			__syncthreads();
			// 由各转动矩阵和原始坐标生成当前lig坐标
			for (int i = threadIdx.x; i < l_atom_numbers; i = i + blockDim.x)
			{
				int current_node_id = l_atom_to_node_serial[i];
				l_frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
				VECTOR temp_crd1 = l_ref_crd[i];
				VECTOR temp_crd2;
				const VECTOR center = l_ref_crd[0];
				while (current_node_id != -1)
				{
					temp_crd2.x = temp_crd1.x - l_node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
					temp_crd2.y = temp_crd1.y - l_node[current_node_id].a0.y;
					temp_crd2.z = temp_crd1.z - l_node[current_node_id].a0.z;

					Matrix_Multiply_Vector(&temp_crd1, l_node[current_node_id].matrix, &temp_crd2);

					temp_crd1.x += l_node[current_node_id].a0.x;
					temp_crd1.y += l_node[current_node_id].a0.y;
					temp_crd1.z += l_node[current_node_id].a0.z;

					current_node_id = l_node[current_node_id].last_node_serial;
				}

				temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
				temp_crd1.y -= center.y;
				temp_crd1.z -= center.z;
				Matrix_Multiply_Vector(&temp_crd2, l_rot_matrix, &temp_crd1);
				l_vina_atom[i].crd.x = temp_crd2.x + l_u_crd[l_u_freedom - 6] + center.x; // 整体平移在最后加上
				l_vina_atom[i].crd.y = temp_crd2.y + l_u_crd[l_u_freedom - 5] + center.y;
				l_vina_atom[i].crd.z = temp_crd2.z + l_u_crd[l_u_freedom - 4] + center.z;
			}
			__syncthreads();
			// 由当前坐标更新l_node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
			for (int node_id = threadIdx.x; node_id < l_node_numbers; node_id = node_id + blockDim.x)
			{
				float temp_length;
				VECTOR tempa, tempn;
				tempa = {l_vina_atom[l_node[node_id].root_atom_serial].crd.x,
						 l_vina_atom[l_node[node_id].root_atom_serial].crd.y,
						 l_vina_atom[l_node[node_id].root_atom_serial].crd.z};
				tempn = {l_vina_atom[l_node[node_id].branch_atom_serial].crd.x,
						 l_vina_atom[l_node[node_id].branch_atom_serial].crd.y,
						 l_vina_atom[l_node[node_id].branch_atom_serial].crd.z};
				tempn.x -= tempa.x;
				tempn.y -= tempa.y;
				tempn.z -= tempa.z;
				temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
				tempn.x *= temp_length;
				tempn.y *= temp_length;
				tempn.z *= temp_length;
				l_node[node_id].n = tempn;
				l_node[node_id].a = tempa;
			}

			__syncthreads(); // 这里实际不需要同步

			float lig_energy_in_thread = 0.f;

			// float lig_fsc_energy_in_thread = 0.f;
			for (int i = threadIdx.x; i < l_atom_numbers; i = i + blockDim.x)
			{
				VINA_ATOM atom_j;
				VECTOR temp_force;
				float rij, dd, dd_, frc_abs, rij_inverse;
				float4 ans;
				int inner_list_start;
				VINA_ATOM atom_i = l_vina_atom[i];
				VECTOR force_i = {0.f, 0.f, 0.f};
				VECTOR dr;
				// 1. Ligand - receptor(+box) interaction
				if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
				{
					// box interaction (transbox)
					dr.x = fdimf(transbox_min.x, atom_i.crd.x); // 如果坐标在盒子外，测提供一个非零矢量，指向盒子内方向
					dr.y = fdimf(transbox_min.y, atom_i.crd.y);
					dr.z = fdimf(transbox_min.z, atom_i.crd.z);
					force_i.x += box_border_strenth * dr.x;
					force_i.y += box_border_strenth * dr.y;
					force_i.z += box_border_strenth * dr.z;
					lig_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

					dr.x = fdimf(atom_i.crd.x, transbox_max.x);
					dr.y = fdimf(atom_i.crd.y, transbox_max.y);
					dr.z = fdimf(atom_i.crd.z, transbox_max.z);
					force_i.x -= box_border_strenth * dr.x;
					force_i.y -= box_border_strenth * dr.y;
					force_i.z -= box_border_strenth * dr.z;
					lig_energy_in_thread += 0.5f * box_border_strenth * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
					// printf(":%f | %f | %f \n", transbox_min.x, atom_i.crd.x, transbox_max.x);
					//  protein interaction
					VECTOR serial; // 在蛋白插值网格中的格点坐标
					serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
					serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
					serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
					ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
					// ans = { 0.f,0.f,0.f,0.f };
					lig_energy_in_thread += ans.w;

					force_i.x += ans.x;
					force_i.y += ans.y;
					force_i.z += ans.z;
				}
				// 2. ligand intra interations
				inner_list_start = i * l_atom_numbers;
				int inner_numbers = l_inner_interaction_list[inner_list_start];
				for (int k = 1; k <= inner_numbers; k = k + 1)
				{
					int j = l_inner_interaction_list[inner_list_start + k];
					atom_j = l_vina_atom[j];
					dr = {atom_i.crd.x - atom_j.crd.x, atom_i.crd.y - atom_j.crd.y, atom_i.crd.z - atom_j.crd.z};
					rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
					if (rij < cutoff)
					{
						float surface_distance = rij - atom_i.radius - atom_j.radius;
						float temp_record;
						// gauss1
						temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
						lig_energy_in_thread += temp_record;
						frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;
						// gauss2
						float dp = surface_distance - k_gauss2_c;
						temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
						lig_energy_in_thread += temp_record;
						frc_abs += 2.f * k_gauss2_2 * temp_record * dp;
						// repulsion
						temp_record = k_repulsion * surface_distance * signbit(surface_distance);
						lig_energy_in_thread += temp_record * surface_distance;
						frc_abs += -2.f * temp_record;
						// hydrophobic
						if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
						{
							temp_record = 1.f * k_hydrophobic;
							lig_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
							frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
						}
						// H bond
						if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
						{
							temp_record = 1.f * k_h_bond;
							lig_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
							frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
						}

						rij_inverse = 1.f / (rij + 10.e-6f);
						frc_abs *= rij_inverse;
						temp_force.x = frc_abs * dr.x;
						temp_force.y = frc_abs * dr.y;
						temp_force.z = frc_abs * dr.z;
						force_i.x += temp_force.x;
						force_i.y += temp_force.y;
						force_i.z += temp_force.z;
						atomicAdd(&l_frc[j].x, -temp_force.x);
						atomicAdd(&l_frc[j].y, -temp_force.y);
						atomicAdd(&l_frc[j].z, -temp_force.z);
					}
				}
				// 3. Ligand-SideChain interactions;
				for (int j = 0; j < f_atom_numbers; j = j + 1)
				{

					atom_j = f_vina_atom[j];

					dr = {atom_i.crd.x - atom_j.crd.x,
						  atom_i.crd.y - atom_j.crd.y,
						  atom_i.crd.z - atom_j.crd.z};

					rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度

					if (rij < cutoff)
					{
						float surface_distance = rij - atom_i.radius - atom_j.radius;
						float temp_record;

						temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
						lig_energy_in_thread += temp_record;
						frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

						float dp = surface_distance - k_gauss2_c;
						temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
						lig_energy_in_thread += temp_record;
						frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

						temp_record = k_repulsion * surface_distance * signbit(surface_distance);
						lig_energy_in_thread += temp_record * surface_distance;
						frc_abs += -2.f * temp_record;

						if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
						{
							temp_record = 1.f * k_hydrophobic;
							lig_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
							frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
						}

						if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
						{
							temp_record = 1.f * k_h_bond;
							lig_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
							frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
						}

						rij_inverse = 1.f / (rij + 10.e-6f);
						frc_abs *= rij_inverse;
						temp_force.x = frc_abs * dr.x;
						temp_force.y = frc_abs * dr.y;
						temp_force.z = frc_abs * dr.z;
						force_i.x += temp_force.x;
						force_i.y += temp_force.y;
						force_i.z += temp_force.z;
						// NO atomicAdd
						// printf("%f\n", lig_energy_in_thread);
					}
					// printf("%d %d %.3f \n" ,i,j,rij);
				}
				atomicAdd(&l_frc[i].x, force_i.x);
				atomicAdd(&l_frc[i].y, force_i.y);
				atomicAdd(&l_frc[i].z, force_i.z);
			}
			// printf("lig_fsc_energy_in_thread: %f\n", lig_fsc_energy_in_thread);
			atomicAdd(&l_shared_data[0], lig_energy_in_thread);
			__syncthreads(); // 能量加和完全，且梯度以及node的叉乘相关信息完全
			if (threadIdx.x == 0)
			{
				l_energy[0] = l_shared_data[0]; // ligand energy
				l_shared_data[1] = l_shared_data[0];
				l_shared_data[0] = 0.f;
				l_alpha1[0] = 0.f;
				l_alpha1[1] = 0.f;
				l_alpha2[0] = 0.f;
				l_alpha2[1] = 0.f;
				l_alpha3[0] = 0.f;
				l_alpha3[1] = 0.f;
			}
			// 计算ligand广义力
			for (int i = threadIdx.x; i < l_atom_numbers; i = i + blockDim.x)
			{
				VECTOR center = {l_vina_atom[0].crd.x, l_vina_atom[0].crd.y, l_vina_atom[0].crd.z};
				VECTOR temp_crd2 = {l_vina_atom[i].crd.x, l_vina_atom[i].crd.y, l_vina_atom[i].crd.z};
				VECTOR temp_crd = temp_crd2;
				VECTOR temp_frc = l_frc[i];
				VECTOR cross;
				VECTOR rot_axis;

				temp_crd.x = temp_crd2.x - center.x;
				temp_crd.y = temp_crd2.y - center.y;
				temp_crd.z = temp_crd2.z - center.z;

				atomicAdd(&l_dU_du_crd[l_u_freedom - 1], (temp_frc.y * temp_crd.z - temp_frc.z * temp_crd.y));
				atomicAdd(&l_dU_du_crd[l_u_freedom - 2], (-temp_frc.x * (temp_crd.y * l_shared_data[14] + temp_crd.z * l_shared_data[13]) + temp_frc.y * temp_crd.x * l_shared_data[14] + temp_frc.z * temp_crd.x * l_shared_data[13]));
				atomicAdd(&l_dU_du_crd[l_u_freedom - 3], (temp_frc.x * (temp_crd.y * l_shared_data[15] - temp_crd.z * l_shared_data[16]) - temp_frc.y * (temp_crd.x * l_shared_data[15] + temp_crd.z * l_shared_data[12]) + temp_frc.z * (temp_crd.x * l_shared_data[16] + temp_crd.y * l_shared_data[12])));

				atomicAdd(&l_dU_du_crd[l_u_freedom - 6], temp_frc.x);
				atomicAdd(&l_dU_du_crd[l_u_freedom - 5], temp_frc.y);
				atomicAdd(&l_dU_du_crd[l_u_freedom - 4], temp_frc.z);

				int current_node_id = l_atom_to_node_serial[i];
				while (current_node_id != -1)
				{
					temp_crd.x = temp_crd2.x - l_node[current_node_id].a.x;
					temp_crd.y = temp_crd2.y - l_node[current_node_id].a.y;
					temp_crd.z = temp_crd2.z - l_node[current_node_id].a.z;
					rot_axis = l_node[current_node_id].n;

					cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
					cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
					cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

					atomicAdd(&l_dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
					current_node_id = l_node[current_node_id].last_node_serial;
				}
			}
			__syncthreads();
			// 进行BB优化更新ligand(暂时未区分整体转动、平动和二面角自由度的各自优化)
			for (int i = threadIdx.x; i < l_u_freedom; i = i + blockDim.x)
			{
				float s = l_u_crd[i] - l_last_u_crd[i];
				float y = l_dU_du_crd[i] - l_last_dU_du_crd[i];
				l_last_u_crd[i] = l_u_crd[i];
				l_last_dU_du_crd[i] = l_dU_du_crd[i];
				if (i < l_u_freedom - 6)
				{
					atomicAdd(&l_alpha1[0], y * s);
					atomicAdd(&l_alpha1[1], y * y);
				}
				else if (i < l_u_freedom - 3)
				{
					atomicAdd(&l_alpha2[0], y * s);
					atomicAdd(&l_alpha2[1], y * y);
				}
				else
				{
					atomicAdd(&l_alpha3[0], y * s);
					atomicAdd(&l_alpha3[1], y * y);
				}
			}
			__syncthreads();

			for (int i = threadIdx.x; i < l_u_freedom; i = i + blockDim.x)
			{
				float du;
				if (i < l_u_freedom - 6)
				{
					float temp_alpha = fabsf(l_alpha1[0]) / fmaxf(l_alpha1[1], 1.e-6f);
					du = temp_alpha * l_dU_du_crd[i];
					du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
				}
				else if (i < l_u_freedom - 3)
				{
					float temp_alpha = fabsf(l_alpha2[0]) / fmaxf(l_alpha2[1], 1.e-6f);
					du = temp_alpha * l_dU_du_crd[i];
					du = copysignf(fmaxf(fabsf(du), 1.f / 10000.f), du);
				}
				else
				{
					float temp_alpha = fabsf(l_alpha3[0]) / fmaxf(l_alpha3[1], 1.e-6f);
					du = temp_alpha * l_dU_du_crd[i];
					du = copysignf(fmaxf(fabsf(du), 2.f * 3.141592654f / 100000.f), du);
				}
				l_dU_du_crd[i] = 0.f;
				l_u_crd[i] += du; // temp muted
			}
			__syncthreads();

			// ligand end
		}
		__syncthreads();
		for (int fsc_opt_i = 0; fsc_opt_i < MAX_OPTIM_STEP_PER_TURN; fsc_opt_i++)
		{
			// 在当前广义坐标下更新fsc各转动矩阵
			for (int i = threadIdx.x; i <= f_node_numbers; i = i + blockDim.x)
			{
				if (i != f_node_numbers)
				{
					float temp_matrix_1[9];
					float cosa, sina, cosa_1;
					sincosf(f_u_crd[i], &sina, &cosa);
					cosa_1 = 1.f - cosa;
					VECTOR temp_n0 = f_node[i].n0;
					temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
					temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
					temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
					temp_matrix_1[3] = temp_matrix_1[1];
					temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
					temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
					temp_matrix_1[6] = temp_matrix_1[2];
					temp_matrix_1[7] = temp_matrix_1[5];
					temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

					f_node[i].matrix[0] = temp_matrix_1[0];
					f_node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
					f_node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
					f_node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
					f_node[i].matrix[4] = temp_matrix_1[4];
					f_node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
					f_node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
					f_node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
					f_node[i].matrix[8] = temp_matrix_1[8];
				}
				else
				{
					float cos_c;
					float sin_c;
					float cos_b;
					float sin_b;
					float cos_a;
					float sin_a;
					sincosf(f_u_crd[f_u_freedom - 3], &sin_c, &cos_c);
					sincosf(f_u_crd[f_u_freedom - 2], &sin_b, &cos_b);
					sincosf(f_u_crd[f_u_freedom - 1], &sin_a, &cos_a);

					f_rot_matrix[0] = cos_b * cos_c;
					f_rot_matrix[1] = cos_b * sin_c;
					f_rot_matrix[2] = -sin_b;
					f_rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
					f_rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
					f_rot_matrix[5] = cos_b * sin_a;
					f_rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
					f_rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
					f_rot_matrix[8] = cos_a * cos_b;

					f_shared_data[11] = cos_b;
					f_shared_data[12] = sin_b;
					f_shared_data[13] = cos_a;
					f_shared_data[14] = sin_a;
					f_shared_data[15] = f_rot_matrix[8]; // cacb
					f_shared_data[16] = f_rot_matrix[5]; // cbsa
				}
			}
			__syncthreads();

			// 由各转动矩阵和原始坐标生成当前fsc坐标
			for (int i = threadIdx.x; i < f_atom_numbers; i = i + blockDim.x)
			{
				int current_node_id = f_atom_to_node_serial[i];
				f_frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
				VECTOR temp_crd1 = f_ref_crd[i];
				VECTOR temp_crd2;
				const VECTOR center = f_ref_crd[0];
				while (current_node_id != -1)
				{
					temp_crd2.x = temp_crd1.x - f_node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
					temp_crd2.y = temp_crd1.y - f_node[current_node_id].a0.y;
					temp_crd2.z = temp_crd1.z - f_node[current_node_id].a0.z;

					Matrix_Multiply_Vector(&temp_crd1, f_node[current_node_id].matrix, &temp_crd2);

					temp_crd1.x += f_node[current_node_id].a0.x;
					temp_crd1.y += f_node[current_node_id].a0.y;
					temp_crd1.z += f_node[current_node_id].a0.z;

					current_node_id = f_node[current_node_id].last_node_serial;
				}

				// 删除平移转动：4.24
				f_vina_atom[i].crd.x = temp_crd1.x + f_u_crd[f_u_freedom - 6];
				f_vina_atom[i].crd.y = temp_crd1.y + f_u_crd[f_u_freedom - 5];
				f_vina_atom[i].crd.z = temp_crd1.z + f_u_crd[f_u_freedom - 4];
				// printf("atom %d x = %.3f + %.3f + %.3f\n", i, temp_crd2.x, u_crd[u_freedom - 6], center.x);
			}
			__syncthreads();
			// FIXME 不知道现在是否需要同步

			// 由当前坐标更新f_node的a和n用于计算广义力（但实际顺序也可以在有了原子力后进行）
			for (int node_id = threadIdx.x; node_id < f_node_numbers; node_id = node_id + blockDim.x)
			{
				float temp_length;
				VECTOR tempa, tempn;
				tempa = {f_vina_atom[f_node[node_id].root_atom_serial].crd.x,
						 f_vina_atom[f_node[node_id].root_atom_serial].crd.y,
						 f_vina_atom[f_node[node_id].root_atom_serial].crd.z};
				tempn = {f_vina_atom[f_node[node_id].branch_atom_serial].crd.x,
						 f_vina_atom[f_node[node_id].branch_atom_serial].crd.y,
						 f_vina_atom[f_node[node_id].branch_atom_serial].crd.z};
				tempn.x -= tempa.x;
				tempn.y -= tempa.y;
				tempn.z -= tempa.z;
				temp_length = rnorm3df(tempn.x, tempn.y, tempn.z);
				tempn.x *= temp_length;
				tempn.y *= temp_length;
				tempn.z *= temp_length;
				f_node[node_id].n = tempn;
				f_node[node_id].a = tempa;
			}
			__syncthreads(); // 这里实际不需要同步
			// FIXME 不知道现在是否需要同步

			// 计算原子力和总能量
			float total_energy_in_thread = 0.f;
			float fsc_energy_in_thread = 0.f;
			float fsc_lig_energy_in_thread = 0.f;
			for (int i = threadIdx.x; i < f_atom_numbers; i = i + blockDim.x)
			{
				VINA_ATOM atom_j;
				VECTOR temp_force;
				float rij, dd, dd_, frc_abs, rij_inverse;
				float4 ans;
				int inner_list_start;
				VINA_ATOM atom_i = f_vina_atom[i];
				VECTOR force_i = {0.f, 0.f, 0.f};
				VECTOR dr;

				// 4. sidechain-receptor interaction
				if (atom_i.atom_type < HYDROGEN_ATOM_TYPE_SERIAL) // 要求是非氢原子
				{
					// no box interaction

					VECTOR serial; // 在蛋白插值网格中的格点坐标
					serial.x = (atom_i.crd.x - box_min.x) * protein_mesh_grid_length_inverse.x;
					serial.y = (atom_i.crd.y - box_min.y) * protein_mesh_grid_length_inverse.y;
					serial.z = (atom_i.crd.z - box_min.z) * protein_mesh_grid_length_inverse.z;
					ans = tex3D<float4>(protein_mesh[atom_i.atom_type], serial.x + 0.5f, serial.y + 0.5f, serial.z + 0.5f); // 自动插值，需要偏离半个格子
					// ans = { 0.f,0.f,0.f,0.f };
					fsc_energy_in_thread += flex_ratio * ans.w;
					force_i.x += flex_ratio * ans.x;
					force_i.y += flex_ratio * ans.y;
					force_i.z += flex_ratio * ans.z;

					// printf("atom #%d protein interaction: %.3f\n", i, ans.w);
				}
				inner_list_start = i * f_atom_numbers;
				int inner_numbers = f_inner_interaction_list[inner_list_start];
				// 5. fsc-fsc interaction
				for (int k = 1; k <= inner_numbers; k = k + 1)
				{
					int j = f_inner_interaction_list[inner_list_start + k];
					atom_j = f_vina_atom[j];

					dr = {atom_i.crd.x - atom_j.crd.x, atom_i.crd.y - atom_j.crd.y, atom_i.crd.z - atom_j.crd.z};
					rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度
					if (rij < cutoff)
					{
						float surface_distance = rij - atom_i.radius - atom_j.radius;
						float temp_record;

						temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
						fsc_energy_in_thread += flex_ratio * temp_record;
						frc_abs = flex_ratio * 2.f * k_gauss1_2 * temp_record * surface_distance;

						float dp = surface_distance - k_gauss2_c;
						temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
						fsc_energy_in_thread += flex_ratio * temp_record;
						frc_abs += flex_ratio * 2.f * k_gauss2_2 * temp_record * dp;

						temp_record = k_repulsion * surface_distance * signbit(surface_distance);
						fsc_energy_in_thread += flex_ratio * temp_record * surface_distance;
						frc_abs += flex_ratio * (-2.f) * temp_record;

						if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
						{
							temp_record = 1.f * k_hydrophobic;
							fsc_energy_in_thread += flex_ratio * temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
							frc_abs += flex_ratio * (-temp_record) * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
						}

						if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
						{
							temp_record = 1.f * k_h_bond;
							fsc_energy_in_thread += flex_ratio * temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
							frc_abs += flex_ratio * (-temp_record) * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
						}

						rij_inverse = 1.f / (rij + 10.e-6f);
						frc_abs *= rij_inverse;
						temp_force.x = frc_abs * dr.x;
						temp_force.y = frc_abs * dr.y;
						temp_force.z = frc_abs * dr.z;
						force_i.x += temp_force.x;
						force_i.y += temp_force.y;
						force_i.z += temp_force.z;
						atomicAdd(&f_frc[j].x, -temp_force.x);
						atomicAdd(&f_frc[j].y, -temp_force.y);
						atomicAdd(&f_frc[j].z, -temp_force.z);
					}
				}
				// 3.' Ligand-SideChain Interactions

				for (int j = 0; j < l_atom_numbers; j = j + 1)
				{
					atom_j = l_vina_atom[j];

					dr = {atom_i.crd.x - atom_j.crd.x,
						  atom_i.crd.y - atom_j.crd.y,
						  atom_i.crd.z - atom_j.crd.z};
					rij = norm3df(dr.x, dr.y, dr.z); // 矢量长度

					if (rij < cutoff)
					{
						float surface_distance = rij - atom_i.radius - atom_j.radius;
						float temp_record;

						temp_record = k_gauss1 * expf(-k_gauss1_2 * surface_distance * surface_distance);
						fsc_lig_energy_in_thread += temp_record;
						// fsc_energy_in_thread += temp_record;
						frc_abs = 2.f * k_gauss1_2 * temp_record * surface_distance;

						float dp = surface_distance - k_gauss2_c;
						temp_record = k_gauss2 * expf(-k_gauss2_2 * dp * dp);
						fsc_lig_energy_in_thread += temp_record;
						// fsc_energy_in_thread += temp_record;
						frc_abs += 2.f * k_gauss2_2 * temp_record * dp;

						temp_record = k_repulsion * surface_distance * signbit(surface_distance);
						fsc_lig_energy_in_thread += temp_record * surface_distance;
						// fsc_energy_in_thread += temp_record * surface_distance;
						frc_abs += -2.f * temp_record;

						if ((atom_i.is_hydrophobic & atom_j.is_hydrophobic))
						{
							temp_record = 1.f * k_hydrophobic;
							fsc_lig_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
							// fsc_energy_in_thread += temp_record * (k_hydrophobic_ua * signbit(surface_distance - k_hydrophobic_a) + k_hydrophobic_ub * signbit(k_hydrophobic_b - surface_distance) + (((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * (surface_distance - k_hydrophobic_a) + k_hydrophobic_ua) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b));
							frc_abs += -temp_record * ((k_hydrophobic_ub - k_hydrophobic_ua) / (k_hydrophobic_b - k_hydrophobic_a)) * signbit(k_hydrophobic_a - surface_distance) * signbit(surface_distance - k_hydrophobic_b);
						}

						if (((atom_i.is_donor & atom_j.is_acceptor) | (atom_i.is_acceptor & atom_j.is_donor)))
						{
							temp_record = 1.f * k_h_bond;
							fsc_lig_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
							// fsc_energy_in_thread += temp_record * (k_h_bond_ua * signbit(surface_distance - k_h_bond_a) + k_h_bond_ub * signbit(k_h_bond_b - surface_distance) + (((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * (surface_distance - k_h_bond_a) + k_h_bond_ua) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b));
							frc_abs += -temp_record * ((k_h_bond_ub - k_h_bond_ua) / (k_h_bond_b - k_h_bond_a)) * signbit(k_h_bond_a - surface_distance) * signbit(surface_distance - k_h_bond_b);
						}
						rij_inverse = 1.f / (rij + 10.e-6f);
						frc_abs *= rij_inverse;
						temp_force.x = frc_abs * dr.x;
						temp_force.y = frc_abs * dr.y;
						temp_force.z = frc_abs * dr.z;
						force_i.x += temp_force.x;
						force_i.y += temp_force.y;
						force_i.z += temp_force.z;
						// NO atomicAdd
					}

					// printf("sc%d,lig%d = %.3f  ; E = %.3f\n" ,i,j,rij,fsc_energy_in_thread);
				}

				atomicAdd(&f_frc[i].x, force_i.x);
				atomicAdd(&f_frc[i].y, force_i.y);
				atomicAdd(&f_frc[i].z, force_i.z);
			}
			atomicAdd(&f_shared_data[0], fsc_energy_in_thread);		 // fsc.energy = fsc-rec + fsc-fsc (intra prot)
			atomicAdd(&f_shared_data[20], fsc_lig_energy_in_thread); // fsc.inter_energy = fsc-lig (inter)
			__syncthreads();										 // 能量加和完全，且梯度以及node的叉乘相关信息完全
			if (threadIdx.x == 0)
			{
				f_energy[0] = f_shared_data[0]; // inter ligand-fsc energy
				f_shared_data[1] = f_shared_data[0];
				f_shared_data[0] = 0.f;
				f_alpha1[0] = 0.f;
				f_alpha1[1] = 0.f;
				inter_energy[0] = f_shared_data[20];
				f_shared_data[21] = f_shared_data[20];
				f_shared_data[20] = 0.f;
			}
			// 计算SideChain广义力
			for (int i = threadIdx.x; i < f_atom_numbers; i = i + blockDim.x)
			{
				VECTOR center = {f_vina_atom[0].crd.x, f_vina_atom[0].crd.y, f_vina_atom[0].crd.z};
				VECTOR temp_crd2 = {f_vina_atom[i].crd.x, f_vina_atom[i].crd.y, f_vina_atom[i].crd.z};
				VECTOR temp_crd = temp_crd2;
				VECTOR temp_frc = f_frc[i];
				VECTOR cross;
				VECTOR rot_axis;

				int current_node_id = f_atom_to_node_serial[i];
				while (current_node_id != -1)
				{
					temp_crd.x = temp_crd2.x - f_node[current_node_id].a.x;
					temp_crd.y = temp_crd2.y - f_node[current_node_id].a.y;
					temp_crd.z = temp_crd2.z - f_node[current_node_id].a.z;
					rot_axis = f_node[current_node_id].n;

					cross.x = temp_crd.y * rot_axis.z - temp_crd.z * rot_axis.y;
					cross.y = temp_crd.z * rot_axis.x - temp_crd.x * rot_axis.z;
					cross.z = temp_crd.x * rot_axis.y - temp_crd.y * rot_axis.x;

					atomicAdd(&f_dU_du_crd[current_node_id], (temp_frc.x * cross.x + temp_frc.y * cross.y + temp_frc.z * cross.z));
					current_node_id = f_node[current_node_id].last_node_serial;
				}
			}
			__syncthreads();

			// 进行BB优化更新side chains
			// 更新：现在只操作二面角，u_freedom-6
			for (int i = threadIdx.x; i < f_u_freedom - 6; i = i + blockDim.x)
			{
				float s = f_u_crd[i] - f_last_u_crd[i];
				float y = f_dU_du_crd[i] - f_last_dU_du_crd[i];
				f_last_u_crd[i] = f_u_crd[i];
				f_last_dU_du_crd[i] = f_dU_du_crd[i];

				// 只有二面角自由度
				atomicAdd(&f_alpha1[0], y * s);
				atomicAdd(&f_alpha1[1], y * y);
			}
			__syncthreads();

			// 更新：u_freedom-6
			for (int i = threadIdx.x; i < f_u_freedom - 6; i = i + blockDim.x)
			{
				float du;
				// torsion du
				float temp_alpha = fabsf(f_alpha1[0]) / fmaxf(f_alpha1[1], 1.e-6f);
				du = temp_alpha * f_dU_du_crd[i];
				du = copysignf(fmaxf(fminf(fabsf(du), 2.f * 2.f * 3.141592654f), 2.f * 3.141592654f / 100000.f), du);
				f_dU_du_crd[i] = 0.f;

				f_u_crd[i] += du; // no muted;
			}
			__syncthreads();
		}

		// 存储Ligand&SideChain能量
	}
}
__device__ void ligand_ref_crd_update(PARTIAL_RIGID_SMALL_MOLECULE::GPU *ligand_gpu)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	__shared__ float shared_data[23];
	float *rot_matrix = &shared_data[2];

	// 进入主循环前的基本初始化
	for (int i = threadIdx.x; i < ligand_gpu->u_freedom; i = i + blockDim.x)
	{
		ligand_gpu->last_u_crd[i] = ligand_gpu->u_crd[i];
		printf("%f ", ligand_gpu->u_crd[i]);
	}
	// 进入主循环前，先同步
	__syncthreads();

	// 在当前广义坐标下更新各转动矩阵
	for (int i = threadIdx.x; i <= ligand_gpu->node_numbers; i = i + blockDim.x)
	{
		if (i != ligand_gpu->node_numbers)
		{
			float temp_matrix_1[9];
			float cosa, sina, cosa_1;
			sincosf(ligand_gpu->u_crd[i], &sina, &cosa);
			cosa_1 = 1.f - cosa;
			VECTOR temp_n0 = ligand_gpu->node[i].n0;
			temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
			temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
			temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
			temp_matrix_1[3] = temp_matrix_1[1];
			temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
			temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
			temp_matrix_1[6] = temp_matrix_1[2];
			temp_matrix_1[7] = temp_matrix_1[5];
			temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

			ligand_gpu->node[i].matrix[0] = temp_matrix_1[0];
			ligand_gpu->node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
			ligand_gpu->node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
			ligand_gpu->node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
			ligand_gpu->node[i].matrix[4] = temp_matrix_1[4];
			ligand_gpu->node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
			ligand_gpu->node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
			ligand_gpu->node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
			ligand_gpu->node[i].matrix[8] = temp_matrix_1[8];
		}
		else
		{
			float cos_c;
			float sin_c;
			float cos_b;
			float sin_b;
			float cos_a;
			float sin_a;
			sincosf(ligand_gpu->u_crd[ligand_gpu->u_freedom - 3], &sin_c, &cos_c);
			sincosf(ligand_gpu->u_crd[ligand_gpu->u_freedom - 2], &sin_b, &cos_b);
			sincosf(ligand_gpu->u_crd[ligand_gpu->u_freedom - 1], &sin_a, &cos_a);

			rot_matrix[0] = cos_b * cos_c;
			rot_matrix[1] = cos_b * sin_c;
			rot_matrix[2] = -sin_b;
			rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
			rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
			rot_matrix[5] = cos_b * sin_a;
			rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
			rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
			rot_matrix[8] = cos_a * cos_b;

			shared_data[11] = cos_b;
			shared_data[12] = sin_b;
			shared_data[13] = cos_a;
			shared_data[14] = sin_a;
			shared_data[15] = rot_matrix[8]; // cacb
			shared_data[16] = rot_matrix[5]; // cbsa
		}

		__syncthreads();

		// 由各转动矩阵和原始坐标生成当前坐标
		for (int i = threadIdx.x; i < ligand_gpu->atom_numbers; i = i + blockDim.x)
		{
			int current_node_id = ligand_gpu->atom_to_node_serial[i];
			// frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
			VECTOR temp_crd1 = ligand_gpu->ref_crd[i];
			VECTOR temp_crd2;
			const VECTOR center = ligand_gpu->ref_crd[0];
			while (current_node_id != -1)
			{
				temp_crd2.x = temp_crd1.x - ligand_gpu->node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
				temp_crd2.y = temp_crd1.y - ligand_gpu->node[current_node_id].a0.y;
				temp_crd2.z = temp_crd1.z - ligand_gpu->node[current_node_id].a0.z;

				Matrix_Multiply_Vector(&temp_crd1, ligand_gpu->node[current_node_id].matrix, &temp_crd2);

				temp_crd1.x += ligand_gpu->node[current_node_id].a0.x;
				temp_crd1.y += ligand_gpu->node[current_node_id].a0.y;
				temp_crd1.z += ligand_gpu->node[current_node_id].a0.z;

				current_node_id = ligand_gpu->node[current_node_id].last_node_serial;
			}

			temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
			temp_crd1.y -= center.y;
			temp_crd1.z -= center.z;
			Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
			ligand_gpu->d_vina_atom[i].crd.x = temp_crd2.x + ligand_gpu->u_crd[ligand_gpu->u_freedom - 6] + center.x; // 整体平移在最后加上
			ligand_gpu->d_vina_atom[i].crd.y = temp_crd2.y + ligand_gpu->u_crd[ligand_gpu->u_freedom - 5] + center.y;
			ligand_gpu->d_vina_atom[i].crd.z = temp_crd2.z + ligand_gpu->u_crd[ligand_gpu->u_freedom - 4] + center.z;
		}
		__syncthreads();
	}

	printf("crd over\n");
	// store ref_crd = current crd
	for (int i = threadIdx.x; i < ligand_gpu->atom_numbers; i = i + blockDim.x)
	{
		ligand_gpu->ref_crd[i] = ligand_gpu->d_vina_atom[i].crd;
		printf("[%.2f, %.2f, %.2f]\n", ligand_gpu->ref_crd[i].x, ligand_gpu->ref_crd[i].y, ligand_gpu->ref_crd[i].z);
	}
	// change u_crd: all dihedrals
	for (int i = 0; i < ligand_gpu->u_freedom - 6; i = i + 1)
	{
		ligand_gpu->u_crd[i] = 0.f;
	}
	// change u_crd: rotation
	ligand_gpu->u_crd[ligand_gpu->u_freedom - 3] = 0.f;
	ligand_gpu->u_crd[ligand_gpu->u_freedom - 2] = 0.f;
	ligand_gpu->u_crd[ligand_gpu->u_freedom - 1] = 0.f;
}
__global__ void update_structure_ref_inplace(
	const int atom_numbers, const int *atom_to_node_serial,
	VECTOR *ref_crd, VINA_ATOM *vina_atom,
	const int u_freedom, float *u_crd, float *last_u_crd,
	const int node_numbers, NODE *node)
{
	// 为考虑可能的加速，共用且小的浮点信息均放到shared上
	//
	__shared__ float shared_data[23];
	float *rot_matrix = &shared_data[2];

	// 进入主循环前的基本初始化
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		last_u_crd[i] = u_crd[i];
	}
	// 进入主循环前，先同步
	__syncthreads();

	// 在当前广义坐标下更新各转动矩阵
	for (int i = threadIdx.x; i <= node_numbers; i = i + blockDim.x)
	{
		if (i != node_numbers)
		{
			float temp_matrix_1[9];
			float cosa, sina, cosa_1;
			sincosf(u_crd[i], &sina, &cosa);
			cosa_1 = 1.f - cosa;
			VECTOR temp_n0 = node[i].n0;
			temp_matrix_1[0] = cosa_1 * temp_n0.x * temp_n0.x + cosa;
			temp_matrix_1[1] = cosa_1 * temp_n0.x * temp_n0.y;
			temp_matrix_1[2] = cosa_1 * temp_n0.x * temp_n0.z;
			temp_matrix_1[3] = temp_matrix_1[1];
			temp_matrix_1[4] = cosa_1 * temp_n0.y * temp_n0.y + cosa;
			temp_matrix_1[5] = cosa_1 * temp_n0.y * temp_n0.z;
			temp_matrix_1[6] = temp_matrix_1[2];
			temp_matrix_1[7] = temp_matrix_1[5];
			temp_matrix_1[8] = cosa_1 * temp_n0.z * temp_n0.z + cosa;

			node[i].matrix[0] = temp_matrix_1[0];
			node[i].matrix[1] = temp_matrix_1[1] + sina * temp_n0.z;
			node[i].matrix[2] = temp_matrix_1[2] - sina * temp_n0.y;
			node[i].matrix[3] = temp_matrix_1[3] - sina * temp_n0.z;
			node[i].matrix[4] = temp_matrix_1[4];
			node[i].matrix[5] = temp_matrix_1[5] + sina * temp_n0.x;
			node[i].matrix[6] = temp_matrix_1[6] + sina * temp_n0.y;
			node[i].matrix[7] = temp_matrix_1[7] - sina * temp_n0.x;
			node[i].matrix[8] = temp_matrix_1[8];
		}
		else
		{
			float cos_c;
			float sin_c;
			float cos_b;
			float sin_b;
			float cos_a;
			float sin_a;
			sincosf(u_crd[u_freedom - 3], &sin_c, &cos_c);
			sincosf(u_crd[u_freedom - 2], &sin_b, &cos_b);
			sincosf(u_crd[u_freedom - 1], &sin_a, &cos_a);

			rot_matrix[0] = cos_b * cos_c;
			rot_matrix[1] = cos_b * sin_c;
			rot_matrix[2] = -sin_b;
			rot_matrix[3] = cos_c * sin_a * sin_b - cos_a * sin_c;
			rot_matrix[4] = cos_a * cos_c + sin_a * sin_b * sin_c;
			rot_matrix[5] = cos_b * sin_a;
			rot_matrix[6] = cos_a * cos_c * sin_b + sin_a * sin_c;
			rot_matrix[7] = -cos_c * sin_a + cos_a * sin_b * sin_c;
			rot_matrix[8] = cos_a * cos_b;

			shared_data[11] = cos_b;
			shared_data[12] = sin_b;
			shared_data[13] = cos_a;
			shared_data[14] = sin_a;
			shared_data[15] = rot_matrix[8]; // cacb
			shared_data[16] = rot_matrix[5]; // cbsa
		}
	}
	__syncthreads();

	// 由各转动矩阵和原始坐标生成当前坐标
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x)
	{
		int current_node_id = atom_to_node_serial[i];
		// frc[i] = {0.f, 0.f, 0.f}; // 在这里清零frc，减少后续一次同步的需求
		VECTOR temp_crd1 = ref_crd[i];
		VECTOR temp_crd2;
		const VECTOR center = ref_crd[0];
		while (current_node_id != -1)
		{
			temp_crd2.x = temp_crd1.x - node[current_node_id].a0.x; // 这里相当于要求node的a0需要和ref相适配，即选择相同的原点
			temp_crd2.y = temp_crd1.y - node[current_node_id].a0.y;
			temp_crd2.z = temp_crd1.z - node[current_node_id].a0.z;

			Matrix_Multiply_Vector(&temp_crd1, node[current_node_id].matrix, &temp_crd2);

			temp_crd1.x += node[current_node_id].a0.x;
			temp_crd1.y += node[current_node_id].a0.y;
			temp_crd1.z += node[current_node_id].a0.z;

			current_node_id = node[current_node_id].last_node_serial;
		}

		temp_crd1.x -= center.x; // 整体转动的参考原点总是第一个原子（root原子）
		temp_crd1.y -= center.y;
		temp_crd1.z -= center.z;
		Matrix_Multiply_Vector(&temp_crd2, rot_matrix, &temp_crd1);
		vina_atom[i].crd.x = temp_crd2.x + u_crd[u_freedom - 6] + center.x; // 整体平移在最后加上
		vina_atom[i].crd.y = temp_crd2.y + u_crd[u_freedom - 5] + center.y;
		vina_atom[i].crd.z = temp_crd2.z + u_crd[u_freedom - 4] + center.z;
	}
	__syncthreads();

	// UPDATE ref_crd inplace
	for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x) // ref_crd <- current crd
	{
		ref_crd[i].x = vina_atom[i].crd.x - vina_atom[0].crd.x;
		ref_crd[i].y = vina_atom[i].crd.y - vina_atom[0].crd.y;
		ref_crd[i].z = vina_atom[i].crd.z - vina_atom[0].crd.z;

		// printf("{%d} %.2f %.2f %.2f \n", i, ref_crd[i].x, ref_crd[i].y, ref_crd[i].z);
	}
	__syncthreads();
	// UPDATE NODE[] inplace
	for (int i = threadIdx.x; i < node_numbers; i = i + blockDim.x)
	{
		// printf("NODE %d: a0 from %.2f %.2f %.2f \n", i, node[i].a0.x, node[i].a0.y, node[i].a0.z);
		//  h_node is based on node. only do neccessary changing here
		//  root->branch
		int &from_atom = node[i].root_atom_serial;
		int &to_atom = node[i].branch_atom_serial;

		node[i].a0 = ref_crd[from_atom];
		node[i].n0 = {ref_crd[to_atom].x - ref_crd[from_atom].x,
					  ref_crd[to_atom].y - ref_crd[from_atom].y,
					  ref_crd[to_atom].z - ref_crd[from_atom].z};
		// printf("from %d [%.2f,%.2f,%.2f] to %d [%.2f,%.2f,%.2f]\n", from_atom, atoms_crd[from_atom].x, atoms_crd[from_atom].y, atoms_crd[from_atom].z, to_atom, atoms_crd[to_atom].x, atoms_crd[to_atom].y, atoms_crd[to_atom].z);
		float length = 1.f / sqrtf(node[i].n0.x * node[i].n0.x +
								   node[i].n0.y * node[i].n0.y +
								   node[i].n0.z * node[i].n0.z);
		// normalization
		node[i].n0.x *= length;
		node[i].n0.y *= length;
		node[i].n0.z *= length;
		// shift
		node[i].a0.x -= ref_crd[0].x;
		node[i].a0.y -= ref_crd[0].y;
		node[i].a0.z -= ref_crd[0].z;

		node[i].a = node[i].a0;
		node[i].n = node[i].n0;
		// printf("NODE %d: a0 to   %.2f %.2f %.2f \n", i, node[i].a0.x, node[i].a0.y, node[i].a0.z);
	}
	__syncthreads();
	// update u_crd here
	for (int i = threadIdx.x; i < u_freedom; i = i + blockDim.x)
	{
		if (i < u_freedom - 6 || i >= u_freedom - 3)
			u_crd[i] = 0.f;
	}
	__syncthreads();
	return;
}

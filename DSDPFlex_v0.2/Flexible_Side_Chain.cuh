#ifndef FLEXIBLE_SIDE_CHAIN_CUH
#define FLEXIBLE_SIDE_CHAIN_CUH
#include "common.cuh"

// ��Ҫ�ṹ�壬���ǵ�ʵ����Ҫ��ͬʱ������������ʽ��������һ��������Ҫ������ʽ����������vina��ֺ������Ŀ���
// Ϊ��֤��Ч���ڶ�stream����ʱ�������ṹ���pdbqt�ļ���ʼ����������ֱ�Ӹ��ƽṹ��
// Ϊ��֤��Ч���ڶ���pdbqt�ļ�ʱ��ͬʱ������ʾ�������tree����ʽ�������tree
// Ϊ���㣬ԭ�����򱣳ֺ�pdbqtһ�£�����pdbqt�Զ�����tree��˳��ṹ�����û��Ҫ����

struct FLEXIBLE_SIDE_CHAIN
{
	int last_modified_time = 20221110;

	// ���ӵĻ�����Ϣ
	int atom_numbers = 0;
	std::vector<VECTOR> frc;			// ԭ���ϵ���
	std::vector<VECTOR> crd;			// ԭ�ӵ�����
	std::vector<VECTOR> crd_from_pdbqt; // ����pdbqt��ԭ�ӵ�����
	// std::vector<VECTOR>root_crd;
	std::vector<VECTOR> origin_crd; // �׺�ԭ������Ϊ0ʱ�����꣬��������crd
	void Refresh_origin_crd(const VECTOR *new_crd);
	VECTOR move_vec;				// ��¼origin_crd���ԭʼ�����pdbqt��λ��ʸ�����������ʼpdbqt�������Ƚϣ���ʵ��Ϊԭʼ��crd[0]λ�á�
	std::vector<int> atomic_number; // ԭ������
	std::vector<int> atom_mass;		// ԭ����������ʱû������
	std::vector<float> charge;		// ԭ�ӵ�ɣ���ʱû������
	std::vector<int> atom_type;		// ԭ�������������ʵ�����Ǹ�char[4]������pdbqt��˵��ʵ�ʾ��ǵ�77��78�е�str
	std::vector<int> root_no;		// ����root(CA)��ԭ�ӱ��

	std::vector<int> dof_resid; // ���Ӳ�����Ϣ
	std::vector<std::string> dof_residues;
	std::vector<int> dof_chis;
	int num_of_res;
	// ��������ɶȵ����˽ṹ�洢
	struct TREE
	{
		int torsion_numbers = 0; // ��Ч������������볣����node_numbers��ͬ
		std::vector<NODE> node;
		std::vector<int> atom_to_node_serial; // ÿ��ԭ�ӵ�����node����ţ�pdbqt��root�ڵ�ԭ�ӵ�nodeΪ-1��ֻ������ת����ƽ�����ɶȣ�
	};

	TREE pdbqt_tree;
	std::vector<int> is_pure_H_freedom; // ��¼pdbqt tree��ÿ���ڵ��Ƿ��Ӧһ�������ת��������ָ�ǻ��ȼ��ţ���vina�в�����ת�����ɶȣ�
										// Ϊ���㹹��vina tree�������ϢͬʱҲ��¼��pdbqtԭʼ��ԭ���б���Ƭ�ڵ���Ϣ
	TREE vina_tree;

	float num_tor = 0.f; // vina���ʹ�õĲ�������ӦΪ��������ɶ��������������ɶ���0.5��

	// Flexible Chains��ʹ�������ʼ������
	void Initial_From_PDBQT(const char *file_name);
	void Initial_From_PDBQT(const char *file_name,
							std::vector<std::string> residue_list,
							std::vector<int> start_posi,
							std::vector<float> initial_dihedral);
	void Copy_From_FLEXIBLE_SIDE_CHAIN(FLEXIBLE_SIDE_CHAIN *input);

	// gpu���
	struct GPU
	{
		FLEXIBLE_SIDE_CHAIN *flexible_side_chain = NULL;
		int u_freedom = 0;	  // ����vina��pdbqt��ʵ�����ɶȲ�ͬ����������ɶȷ���gpu�ڲ�
		int atom_numbers = 0; // �͸��ṹ��ͬ������Ϣ���������
		int node_numbers = 0;

		int malloced_atom_numbers = 0;

		VECTOR *origin_crd = NULL; // һ����rootԭ������ƽ�Ƶ�(0,0,0)����ԭʼ�������꣬���������������б��ֲ��䣨������ֵ���¡�Ƕ��ռ���Ρ���
		VECTOR *ref_crd = NULL;	   // ʵ������crdʱʹ�õĲο����꣬��������origin_crd��ȣ�һ��ֻ��һ������ת������
		VECTOR *crd = NULL;
		VECTOR *last_crd = NULL; // ���ڱ������о�
		VECTOR *frc = NULL;
		int *atom_to_node_serial = NULL;
		int *inner_neighbor_list = NULL; // С�����ڲ���ԭ�Ӽ��໥���ý��ڱ�������С���ӵ��ձ��С��ʵ�ʾ��൱����������һ�μ��㣬��������Ҫ�޳�һЩԭ�Ӷԣ�1-2��1-3��1-4���ã������ֱ�Ӹ��һ���̶��Ľ��ڱ���������ʵ�ʴ�СΪatom_numbers*atom_numbers��ʵ��Ӧ���������󣬵�Ŀǰ��������
		int *h_inner_neighbor_list = NULL;

		int malloced_u_freedom = 0;
		float *h_u_crd = NULL; // ��������ʼ�����ɶ������Լ�ż�����д�����Ŷ�ǿ�ƽ���
		float *u_crd = NULL;
		float *last_u_crd = NULL;			 // BB opt
		float *h_last_accepted_u_crd = NULL; // MC
		float *dU_du_crd = NULL;
		float *last_dU_du_crd = NULL; // BB opt

		int malloced_node_numbers = 0;
		NODE *node = NULL;

		void Initial(FLEXIBLE_SIDE_CHAIN *mol, const TREE *tree);

		// MC��vina�������
		float last_accepted_energy = 100.f;
		float last_accepted_inter_energy = 100.f;
		VINA_ATOM *h_vina_atom = NULL;
		VINA_ATOM *d_vina_atom = NULL;

		float *inter_energy = &h_u_crd[u_freedom - 1];
		float *intra_energy = &h_u_crd[u_freedom];
	};

	// Ŀǰֻ��ʱ����ʹ��vina�����Ĺ���
	GPU vina_gpu;
	GPU pdbqt_gpu; // ���������H�����ɶȲ���
};

#endif // FLEXIBLE_SIDE_CHAIN_CUH
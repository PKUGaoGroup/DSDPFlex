#include "common.cuh"
#include <unordered_map>
#include <iostream>
#include <random>
#include <set>
struct Rotamer_Entry
{
    std::string res_name;
    float prob;
    float chi[4];
    float sigma[4];
};
struct Residue_nature
{
    int dof;
    struct Torsion_def
    {
        std::string atom_names[5];
    };

    std::vector<Torsion_def> rotamer_def;
};
struct Rotamer
{
    // All angle values are in Rad (0~2Pi), identical to Kernel
    // Degree of Freedom
    int dof;
    std::vector<std::string> residue_list;           // dof[i] -> residue name of i
    std::vector<int> residue_index;                  // dof[i] -> resid of i
    std::vector<int> start_position;                 // dof[i] -> start position of i
    std::vector<float> initial_dihedral;             // initial dihedral calced for each dof
                                                     // FIXME some dof is H dof, not related to torsion
    std::vector<bool> active_torsion;                // TODO 如果用vina_tree则不需要active_torsion
    std::unordered_map<int, VECTOR> backbone_N_crds; // get a backbone N crd by resid
    // Rotamer Library store
    std::vector<Rotamer_Entry> entrys; // temp
    std::unordered_map<std::string, std::vector<Rotamer_Entry>> resname_to_entrys{
        {"ARG", {}},
        {"ASN", {}},
        {"ASP", {}},
        {"CYS", {}},
        {"GLN", {}},
        {"GLU", {}},
        {"HIS", {}},
        {"ILE", {}},
        {"LEU", {}},
        {"LYS", {}},
        {"MET", {}},
        {"PHE", {}},
        {"PRO", {}},
        {"SER", {}},
        {"THR", {}},
        {"TRP", {}},
        {"TYR", {}},
        {"VAL", {}}}; // information from rotamer lib
    ;
    FILE *lib = new FILE();
    FILE *def = new FILE();
    char line[256];
    char resname[64];
    char atom0[64], atom1[64], atom2[64], atom3[64];
    std::unordered_map<std::string, Residue_nature> residue_info{
        {"ARG", {4, {}}},
        {"ASN", {2, {}}},
        {"ASP", {2, {}}},
        {"CYS", {1, {}}},
        {"GLN", {3, {}}},
        {"GLU", {3, {}}},
        {"HIS", {2, {}}},
        {"ILE", {2, {}}},
        {"LEU", {2, {}}},
        {"LYS", {4, {}}},
        {"MET", {3, {}}},
        {"PHE", {2, {}}},
        {"PRO", {2, {}}},
        {"SER", {1, {}}},
        {"THR", {1, {}}},
        {"TRP", {2, {}}},
        {"TYR", {2, {}}},
        {"VAL", {1, {}}}};
    void Initialize_def(const char *defpath);
    void Initialize_lib(const char *libpath);
    void Initialize(const char *defpath, const char *libpath)
    {
        this->Initialize_def(defpath);
        this->Initialize_lib(libpath);
    }
    void Initial_from_residue_list(); // when residue_list is made

    /*
     * Sample from rotamer lib
     * result located in float *out
     * Return the dof of this `resname' (aka useful result number)
     */
    int do_sampling(std::string resname, float *out);
    void Get_backbone_N(std::vector<VECTOR_INT> *backbone_Ns, VECTOR shift);
    void Get_initial_dihedral(std::vector<NODE> tree_node);
    void Initial_dihedral_from_PDBQT(char *rigid_file, char *flex_file);
    void Torsion_sample(float *torsions_u, int flex_dof);     // random sample one rotamer
    void Torsion_initialize(float *torsions_u, int flex_dof); // randomize all residues
};

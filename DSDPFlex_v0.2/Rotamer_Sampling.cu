#include "Rotamer_Sampling.cuh"
void Rotamer::Initialize_def(const char *defpath)
{
    def = fopen(defpath, "r");
    if (!def)
    {
        throw std::runtime_error("Failed to initialize rotamer library.");
    }
    while (fgets(line, 256, def))
    {
        if (line[0] == '#' || line[0] == ' ' || line[0] == '\n')
            continue;

        sscanf(line, "%s %s %s %s %s ", resname, atom0, atom1, atom2, atom3);
        // read line N CA CB CG
        Residue_nature::Torsion_def temp_rotamer_def = {
            std::string(atom0),
            std::string(atom1),
            std::string(atom2),
            std::string(atom3)};

        this->residue_info
            .at(std::string(resname))
            .rotamer_def.push_back(temp_rotamer_def);
    }
    fclose(def);
}
void Rotamer::Initialize_lib(const char *libpath)
{

    lib = fopen(libpath, "r");
    if (!lib)
    {
        throw std::runtime_error("Failed to initialize rotamer library.");
    }
    // Read rotamer lib

    while (fgets(line, 256, lib))
    {
        // read from Dunbrack Rotamer Library format
        if (line[0] == '#' || line[0] == ' ' || line[0] == '\n')
            continue;

        int t1, t2, t3, t4, tcntr1, tcrot;
        float p1234, sig_p1234, p234_1, sig_p234_1, chi1 = 1e10, sig_chi1, chi2 = 1e10, sig_chi2, chi3 = 1e10, sig_chi3, chi4 = 1e10, sig_chi4;

        char resname[3] = {line[0], line[1], line[2]};
        sscanf(line, "%s ", resname);
        int tordof = residue_info.at(std::string(resname)).dof;
        switch (tordof)
        {
        case 4:
            sscanf(line, "%s %d %d %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f",
                   resname, &t1, &t2, &t3, &t4, &tcntr1, &tcrot,
                   &p1234, &sig_p1234, &p234_1, &sig_p234_1,
                   &chi1, &sig_chi1, &chi2, &sig_chi2, &chi3, &sig_chi3, &chi4, &sig_chi4);
            break;
        case 3:
            sscanf(line, "%s %d %d %d %d %d %d %f %f %f %f %f %f %f %f %f %f",
                   resname, &t1, &t2, &t3, &t4, &tcntr1, &tcrot,
                   &p1234, &sig_p1234, &p234_1, &sig_p234_1,
                   &chi1, &sig_chi1, &chi2, &sig_chi2, &chi3, &sig_chi3);
            break;
        case 2:
            sscanf(line, "%s %d %d %d %d %d %d %f %f %f %f %f %f %f %f",
                   resname, &t1, &t2, &t3, &t4, &tcntr1, &tcrot,
                   &p1234, &sig_p1234, &p234_1, &sig_p234_1,
                   &chi1, &sig_chi1, &chi2, &sig_chi2);
            break;
        case 1:
            sscanf(line, "%s %d %d %d %d %d %d %f %f %f %f %f %f",
                   resname, &t1, &t2, &t3, &t4, &tcntr1, &tcrot,
                   &p1234, &sig_p1234, &p234_1, &sig_p234_1,
                   &chi1, &sig_chi1);
            break;
        default:
            break;
        }
        Rotamer_Entry en = {resname, p1234, {chi1, chi2, chi3, chi4}, {sig_chi1, sig_chi2, sig_chi3, sig_chi4}};

        for (int i = 0; i < 4; i++)
        {
            // to 0~360
            if (en.chi[i] < 0)
                en.chi[i] += 360;
            // to rad
            en.chi[i] *= M_PI / 180;
            en.sigma[i] *= M_PI / 180;
        }
        // printf("%s\n", resname);
        // printf("%d\n", t1);
        this->resname_to_entrys.at(std::string(resname)).push_back(en); // store information
    }
    fclose(lib);
    // for (auto i = resname_to_entrys.begin(); i != resname_to_entrys.end(); i++)
    // {
    //     std::vector<Rotamer_Entry> tmp = i->second;
    //     std::cout << i->first << std::endl;
    //     printf("%d", tmp.size());
    //     for (int j = 0; j < tmp.size(); j++)
    //     {
    //         std::cout << tmp[j].resname << tmp[j].prob << tmp[j].chi1 << tmp[j].chi2 << tmp[j].chi3 << tmp[j].chi4 << std::endl;
    //     }
    // }
}
void Rotamer::Initial_from_residue_list() // residue_list has been made
{
    // this->residue_list = reslist;
    // zhubibaba, can't use res name!!!, use res id!!
    std::string current_resn{residue_list[0]};
    int current_resi{residue_index[0]};
    start_position.clear();
    int current_start = 0; // this RES start at ?
    int last_res_size = 0;

    for (int i = 0; i < this->residue_list.size(); i++)
    {
        std::string this_resn = this->residue_list[i];
        int this_resi = this->residue_index[i];

        if (current_resn != this_resn || current_resi != this_resi) // a new RES
        {
            // do check last res: now, current=last
            last_res_size = i - current_start;
            int proper_size_of_last_res = this->residue_info.at(current_resn).dof;
            if (last_res_size != proper_size_of_last_res)
            {
                // Warning about
                // Maybe because H
                // printf("WARNING: Rotamer found inconsistence:\n"
                //        "  Residue %s should have %d degree(s) of freedom, %d found in flex PDBQT\n"
                //        "  Assuming pure H freedom\n",
                //        current_resn.c_str(),
                //        proper_size_of_last_res, last_res_size);
                this->active_torsion[i - 1] = false;
            }
            current_resn = this_resn;
            current_resi = this_resi;
            current_start = i;
        }
        // res didn't change, so 'start_position' same
        this->start_position.push_back(current_start);
        this->active_torsion.push_back(true);
    }

    // check    printf("ok returned\n");
    if (start_position.size() != residue_list.size())
    {
        printf("%d %d\n", start_position.size(), residue_list.size());
        return;
    }
    // printf("check start point:\n");
    // for (int j = 0; j < residue_index.size(); j++)
    // {
    //     printf("%s %d start at %d\n", residue_list[j].c_str(), residue_index[j], start_position[j]);
    // }
}
/*
 * Sample from rotamer lib
 * result located in float *out
 * Return the dof of this `resname' (aka useful result number)
 */
int Rotamer::do_sampling(std::string resname, float *out)
{

    std::vector<Rotamer_Entry> rotamers = this->resname_to_entrys.at(resname);
    int tordof = residue_info.at(resname).dof;
    // printf("resname = %s ; tordof = %d\n", resname.c_str(), tordof);
    unsigned seed = rand();
    std::default_random_engine gen(seed);
    float f = ((float)rand()) / RAND_MAX * 100; // prob sample, must convert to float!!

    // sampling method
    for (int i = 0; i < rotamers.size(); i++)
    {
        f -= rotamers[i].prob;

        if (f < 0)
        {
            // printf("|%f %f %f %f\n", rotamers[i].chi[0], rotamers[i].chi[1], rotamers[i].chi[2], rotamers[i].chi[3]);
            for (int j = 0; j < tordof; j++)
            {
                out[j] = std::normal_distribution<float>(rotamers[i].chi[j], rotamers[i].sigma[j])(gen);
            }
            return tordof;
        }
    }
    // if no return
    for (int j = 0; j < tordof; j++)
    {
        out[j] = 2 * M_PI * rand() / RAND_MAX; // simple random conformation
    }
    return tordof;
}
void Rotamer::Get_backbone_N(std::vector<VECTOR_INT> *backbone_Ns, VECTOR shift)
{
    // flex reference crd

    // optimized
    std::set<int>
        resid_set(residue_index.begin(), residue_index.end());
    int resid_max = *std::max_element(this->residue_index.begin(), this->residue_index.end());
    int resid_min = *std::min_element(this->residue_index.begin(), this->residue_index.end());

    for (auto bb_N = (*backbone_Ns).begin(); bb_N != (*backbone_Ns).end(); bb_N++)
    {
        VECTOR_INT this_N = *bb_N;
        if (this_N.type < resid_min - 1)
            continue;
        if (this_N.type > resid_max + 1)
            break;
        if (resid_set.find(this_N.type) != resid_set.end())
        { // if found in residue id
            this->backbone_N_crds.insert({this_N.type,
                                          VECTOR{this_N.x - shift.x,
                                                 this_N.y - shift.y,
                                                 this_N.z - shift.z}});
            // printf("%d %f %f %f\n", this_N.type, this_N.x, this_N.y, this_N.z);
        }
    }
}
void Rotamer::Get_initial_dihedral(std::vector<NODE> tree_node)
{
    // from NODE get a0
    // if NODE->-1 (root), find a backbone N
    for (int i = 0; i < tree_node.size(); i++)
    {
        std::string resn = this->residue_list[i];
        int resi = this->residue_index[i];
        NODE the_node = tree_node[i]; // current node
        printf("%d NODE: last serial %d\n", i, the_node.last_node_serial);
        print_vec(the_node.a0, "v");
        VECTOR v_ab, v_bc, v_cd;             // used to calc dihedral
        if (the_node.last_node_serial == -1) // -1 == root == Ns
        {
            // root node
            VECTOR bb_N = backbone_N_crds.at(resi);
            print_vec(bb_N, "N");
        }
    }
}

inline VECTOR _get_atom_crd(char *str_line)
{
    // get atom crd
    VECTOR temp_crd;
    char temp_float_str[9];
    temp_float_str[8] = '\0';
    for (int i = 0; i < 8; i = i + 1)
    {
        temp_float_str[i] = str_line[30 + i];
    }
    sscanf(temp_float_str, "%f", &temp_crd.x);
    for (int i = 0; i < 8; i = i + 1)
    {
        temp_float_str[i] = str_line[38 + i];
    }
    sscanf(temp_float_str, "%f", &temp_crd.y);
    for (int i = 0; i < 8; i = i + 1)
    {
        temp_float_str[i] = str_line[46 + i];
    }
    sscanf(temp_float_str, "%f", &temp_crd.z);
    return temp_crd;
}
/* Directly calculat dihedral from input pdbqt(s) */
void Rotamer::Initial_dihedral_from_PDBQT(char *rigid_file, char *flex_file)
{

    FILE *rigid = fopen_safely(rigid_file, "r");
    FILE *flex = fopen_safely(flex_file, "r");
    char str_line[256];
    char str_seg[256];

    std::set<int>
        resid_set(residue_index.begin(), residue_index.end());

    /* Read rigid file to get backbone N */
    while (fgets(str_line, 256, rigid))
    {
        sscanf(str_line, "%s", str_seg);

        if (strcmp(str_seg, "ATOM") == 0)
        {

            int resi = 0; // residue ID
            sscanf(&str_line[22], "%d", &resi);

            // if this resi is required & this is N
            if (resid_set.find(resi) != resid_set.end() &&
                str_line[13] == 'N' &&
                str_line[14] == ' ')
            {
                // get atom crd
                VECTOR temp_crd = _get_atom_crd(str_line);
                this->backbone_N_crds.insert({resi, temp_crd});
            }
        }
    }
    // printf("read rigid ok\n");
    /* Read flex file to get CA-CB-CG-CD.. */
    std::unordered_map<std::string, VECTOR> atom_name_to_crd;
    std::vector<std::unordered_map<std::string, VECTOR>> list_atom_to_crd; // {{res1_atom1:crd, res1_atom2:crd, ...}, {res2}, ...}
    std::vector<std::string> resn_todo;

    int current_handling_resi = 0;
    std::string current_handling_resn = {};
    int last_handled_resi = 0;
    bool crd_loaded = false;

    // printf("read all ok\n");
    while (fgets(str_line, 256, flex))
    {
        sscanf(str_line, "%s", str_seg);

        if (strcmp(str_seg, "ATOM") == 0 && str_line[77] != 'H') // heavy atoms, actually, H never involved in rotamers
        {
            int resi = 0;                       // residue ID
            sscanf(&str_line[22], "%d", &resi); // current resi
            std::string resn = {str_line[17],
                                str_line[18],
                                str_line[19]};
            if (resi != current_handling_resi)
            {
                // end of inputting a resi

                if (crd_loaded)
                {
                    // printf("%d\n", current_handling_resi);
                    VECTOR N_crd = this->backbone_N_crds.at(current_handling_resi);
                    atom_name_to_crd.insert({"N", N_crd});                                          // N used in this res
                    list_atom_to_crd.push_back({atom_name_to_crd.begin(), atom_name_to_crd.end()}); // append
                    resn_todo.push_back(current_handling_resn);
                }
                current_handling_resi = resi;
                current_handling_resn = resn;
                atom_name_to_crd.clear(); // last don't have a clear
            }

            // load atom name and crd
            VECTOR temp_crd = _get_atom_crd(str_line);
            std::string atom_name = {str_line[13], str_line[14], str_line[15]};
            atom_name.erase(atom_name.find_last_not_of(" ") + 1);
            atom_name_to_crd.insert({atom_name, temp_crd});
            crd_loaded = true;
        }
    }
    fclose(flex);
    // record the last one
    VECTOR N_crd = this->backbone_N_crds.at(current_handling_resi);
    atom_name_to_crd.insert({"N", N_crd}); // N used in this res
    list_atom_to_crd.push_back({atom_name_to_crd.begin(), atom_name_to_crd.end()});
    resn_todo.push_back(current_handling_resn);

    // printf("check %d %d", list_atom_to_crd.size(), resn_todo.size());
    // for (int i = 0; i < list_atom_to_crd.size(); i++)
    // {
    //     auto map1 = list_atom_to_crd[i];
    //     printf("#%d %d ", i, map1.size());
    //     for (auto ii = map1.begin(); ii != map1.end(); ii++)
    //     {
    //         printf("%s (%d) ", (*ii).first.c_str(), (*ii).first.length());
    //     }
    //     printf("\n");
    // }

    // where to put dihedrals
    std::vector<float> temp_dihedrals;
    std::vector<int> startp = {};
    int curr_start = -1;

    for (int i = 0; i < residue_index.size(); i++)
    {
        int rstart = this->start_position[i];
        if (rstart != curr_start)
        {
            startp.push_back(rstart);
            curr_start = rstart;
        }
    }
    // check `residue numbers'
    if (startp.size() != resn_todo.size() ||
        startp.size() != list_atom_to_crd.size())
    {

        throw std::runtime_error("Failed to calculate flex initial dihedrals");
    }

    this->initial_dihedral.resize(this->residue_list.size());                     // same as total dof
    std::fill(this->initial_dihedral.begin(), this->initial_dihedral.end(), 0.f); // set to 0
                                                                                  // start calculation, for each residue do
    printf("initial dihedrals\n"
           "name |    chi def   | degree\n");
    for (int i = 0; i < list_atom_to_crd.size(); i++)
    {
        auto atom_to_crd = list_atom_to_crd[i];
        auto the_resn = resn_todo[i];

        // get rotamers' definition
        auto resn_rotamer_def = this->residue_info.at(the_resn).rotamer_def;

        // for k-th torsion angle, calc
        printf("%s%d\n", the_resn.c_str(), residue_index[startp[i]]);
        for (int k = 0; k < resn_rotamer_def.size(); k++)
        {
            auto irot = resn_rotamer_def[k];
            std::vector<VECTOR> crd4; // 4 atoms, 3 vectors N CA CB CG
            for (int l = 0; l < 4; l++)
            {
                // printf("ATOM NAME %s (%d)\n", irot->atom_names[k].c_str(), irot->atom_names[k].length());
                crd4.push_back(atom_to_crd.at(irot.atom_names[l]));
            }
            VECTOR vec3[3];
            for (int l = 0; l < 3; l++)
            {
                vec3[l] = {crd4[l + 1].x - crd4[l].x,
                           crd4[l + 1].y - crd4[l].y,
                           crd4[l + 1].z - crd4[l].z};
            }
            float dihedral = calc_dihedral(vec3[0], vec3[1], vec3[2]);
            // should output standard dihedral

            printf("  %d  %3s-%3s-%3s-%3s  %5.1f\n", k + 1,
                   irot.atom_names[0].c_str(),
                   irot.atom_names[1].c_str(),
                   irot.atom_names[2].c_str(),
                   irot.atom_names[3].c_str(),
                   dihedral * 180 / 3.14159265);

            // store value
            int dof_index = startp[i] + k; // i-th res, k-th tors
            if (abs(this->initial_dihedral[dof_index]) > 1e6)
            {
                printf("WARNING: something is wrong at #%d torsion, calculated more than once\n", dof_index);
            }
            else
            {
                this->initial_dihedral[dof_index] = dihedral;
            }
        }
    }

    // this->initial_dihedral;

    // printf("checking initial dihedral \n");
    // int idx = 0;
    // for (auto it = initial_dihedral.begin(); it != initial_dihedral.end(); it++)
    // {
    //     printf("[%d]%f \n", idx, (*it) * 180 * M_1_PI);
    //     idx++;
    // }
    // printf("\n");
}

void Rotamer::Torsion_sample(float *torsions_u, int flex_dof)
{
    // printf("--- 1 sampling happening!! ---\n");
    int sampling_density = 1; // sample one tors
    int rand_select = rand() % flex_dof;
    // printf("previous u\n");
    //  for (int i = 0; i < flex_dof; i++)
    //  {
    //      printf(" %.1f ", torsions_u[i] * 180 * M_1_PI);
    //  }
    //  printf("\n");
    //  printf("select %d ", rand_select);

    int rindex = residue_index[rand_select];
    std::string resname = residue_list[rand_select];
    int u_start = start_position[rand_select]; // u_start, u_start +1, ..., u_start + u_len
    int u_len = 0;
    // do random sampling
    float tors_sampled[5] = {0., 0., 0., 0., 0.};
    u_len = this->do_sampling(resname, tors_sampled);
    // u should be the value of angle(init->rotamer)
    // a.k.a how to move a res from initial to one rotamer
    for (int i = 0; i < u_len; i++)
    {
        // note: angles are periodic
        float v = tors_sampled[i] - this->initial_dihedral[u_start + i];
        if (v < 0)
            v += 2 * M_PI;
        // change u
        torsions_u[u_start + i] = v;
    }
    // printf("current u\n");
    //  for (int i = 0; i < flex_dof; i++)
    //  {
    //      printf(" %.1f ", torsions_u[i] * 180 * M_1_PI);
    //  }
    //  printf("\n");
}
void Rotamer::Torsion_initialize(float *torsions_u, int flex_dof)
{
    // printf("--- 1 initialization happening!! ---\n");
    //  printf("%d", flex_dof);
    int handled_max = -1;
    for (auto it = this->start_position.begin(); it < this->start_position.end(); it++)
    {
        int stp = *it;
        if (stp <= handled_max)
            continue;
        // Test rotamer sampling
        float tors_sampled[5] = {0., 0., 0., 0., 0.};
        std::string curr_residue = this->residue_list[stp];
        int u_len = 0;
        u_len = this->do_sampling(curr_residue, tors_sampled);

        // printf("%s (%d): ", curr_residue.c_str(), u_len);

        // printf("I got %f %f %f %f\n", tors_sampled[0], tors_sampled[1], tors_sampled[2], tors_sampled[3]);

        for (int k = 0; k < u_len; k++)
        {
            float v = tors_sampled[k] - this->initial_dihedral[stp + k];
            if (v < 0)
                v += 2 * M_PI;
            torsions_u[stp + k] = v;
            // printf("%d+%d  %.1f ", stp, k, this->initial_dihedral[stp + k] * 180 * M_1_PI);
        }
        // printf("\n");
        handled_max = stp;
    }
}
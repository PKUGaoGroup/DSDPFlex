#include "DSDP_Sort.cuh"
void DSDP_SORT::Sort_Structures(const int lig_atom_numbers, const int fsc_atom_numbers,
								const int *lig_atomic_number,
								const int record_numbers, const VECTOR *lig_crd_record, const VECTOR *fsc_crd_record,
								const INT_FLOAT *serial_energy_list,
								const float rmsd_cutoff, const int forward_comparing_numbers, const int desired_selecting_numbers)
{
	this->lig_atom_numbers = lig_atom_numbers;
	this->fsc_atom_numbers = fsc_atom_numbers;
	selected_numbers = 0;
	// 不用push_back是为了方便坐标复制
	selected_lig_crd.resize((size_t)desired_selecting_numbers * lig_atom_numbers);
	selected_fsc_crd.resize((size_t)desired_selecting_numbers * fsc_atom_numbers);
	selected_energy.resize(desired_selecting_numbers);

	for (int frame_i = 0; frame_i < record_numbers; frame_i += 1)
	{
		bool is_existing_similar_structure = false;
		for (int pose_i = selected_numbers - 1; pose_i >= 0; pose_i -= 1)
		{
			// only consider ligand
			float rmsd = calcualte_heavy_atom_rmsd(
				lig_atom_numbers,
				&selected_lig_crd[(size_t)pose_i * lig_atom_numbers],
				&lig_crd_record[(size_t)serial_energy_list[frame_i].id * lig_atom_numbers],
				lig_atomic_number);

			if (rmsd < rmsd_cutoff)
			{
				is_existing_similar_structure = true;
				break;
			}
		}
		if (is_existing_similar_structure == false)
		{
			// copy lig crd
			memcpy(&selected_lig_crd[(size_t)selected_numbers * lig_atom_numbers],
				   &lig_crd_record[(size_t)serial_energy_list[frame_i].id * lig_atom_numbers],
				   sizeof(VECTOR) * lig_atom_numbers);
			// copy fsc crd
			memcpy(&selected_fsc_crd[(size_t)selected_numbers * fsc_atom_numbers],
				   &fsc_crd_record[(size_t)serial_energy_list[frame_i].id * fsc_atom_numbers],
				   sizeof(VECTOR) * fsc_atom_numbers);
			selected_energy[selected_numbers] = serial_energy_list[frame_i].energy;
			selected_numbers = selected_numbers + 1;
			if (selected_numbers >= desired_selecting_numbers)
			{
				break;
			}
		}
	}
}

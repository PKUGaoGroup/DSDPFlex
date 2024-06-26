python DSDP_Tools/DSDPFlex.py \
--ligand ./test/136_4TWI_1M2K/Ligand.pdbqt \
--flex ./test/136_4TWI_1M2K/APO_flex.pdbqt \
--protein ./test/136_4TWI_1M2K/APO_rigid.pdbqt \
--box_min -15.43 -16.96 -11.65 \
--box_max 11.51 14.65 23.66  \
--exhaustiveness 384 \
--ligbox_min -12.64 -14.18 5.73 \
--ligbox_max  6.28 11.12 21.05 \
--search_depth 40 \
--top_n 10 \
--out ./test/136_4TWI_1M2K/pyDSDP_out.pdbqt \
--out_flex ./test/136_4TWI_1M2K/pyDSDP_out_flex.pdbqt \
--rescore gnina \
--verbose \
--kernel_type 0
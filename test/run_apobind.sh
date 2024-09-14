
while read line
do a=`awk '{print $1}' ./apobind_prepared/${line}/box.txt`
b=`awk '{print $2}' ./apobind_prepared/${line}/box.txt`
c=`awk '{print $3}' ./apobind_prepared/${line}/box.txt`
d=`awk '{print $4}' ./apobind_prepared/${line}/box.txt`
e=`awk '{print $5}' ./apobind_prepared/${line}/box.txt`
f=`awk '{print $6}' ./apobind_prepared/${line}/box.txt`
g=`awk '{print $7}' ./apobind_prepared/${line}/box.txt`
h=`awk '{print $8}' ./apobind_prepared/${line}/box.txt`
i=`awk '{print $9}' ./apobind_prepared/${line}/box.txt`
j=`awk '{print $10}' ./apobind_prepared/${line}/box.txt`
k=`awk '{print $11}' ./apobind_prepared/${line}/box.txt`
l=`awk '{print $12}' ./apobind_prepared/${line}/box.txt`

../bin/DSDPflex --ligand ./apobind_prepared/${line}/ligand.pdbqt --flex ./apobind_prepared/${line}/receptor_flex.pdbqt --protein ./apobind_prepared/${line}/receptor_rigid.pdbqt --box_min $g $h $i --box_max $j $k $l --ligbox_min $a $b $c --ligbox_max $d $e $f  --top_n 10 --out ./apobind_prepared/${line}/ligand_out.pdbqt --out_flex ./apobind_prepared/${line}/flex_out.pdbqt --log ./apobind_prepared/${line}/dsdp_out.log

done < apobind.txt

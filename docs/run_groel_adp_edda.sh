# download top from CHARMM GUI
# add it to solutes dir: ligandrm.pdb, charmm36.itp that contains the ligand itp, and ADP.itp ?


cp ../0105_EDDA/*.mdp ./
cp ../0105_EDDA/topol.top ./
cp ../0105_EDDA/test.gro ./
cp ../0105_EDDA/*.itp ./
cp ../0105_EDDA/topol_Protein_chain_*.gro ./

gmx editconf -f test.gro -o test_box.gro -c -d 2 -bt cubic

# to mimic 100 equiv of ADP, but buffer are using real conc, ie, 200mM; Mg is 1 mM (but only serve to neutralize the system, not to mimic real conc)
# Charge = protein(-19*7=-133) + ADP(-3*300=-900) + MDA(+1*425=+425) + DDA(+2*425=+850) ACE (-1*1700) = -1458


# protein is 1um
# ADP is 300 um max in EDDA test, using equiv, we insert 300 ADP
# for EDDA=MDA+EDA, it will be too much if we use equiv, so we swithed to real conc
# conc = 200 mM = [ 850/(6.02e23) ] / (19.14e-8)^3, so we insert 425 MDA and 425 DDA
# for ACE, it is 400 mM (EDDA~2AC), which gives 1700 ACE in the box

# the total charge after is
# protein = -19*7=-133
# ADP = -3*300=-900
# MDA = +1*425=+425
# DDA = +2*425=+850
# ACE = -1*1700=-1700
# total = -1458, so we need 729 Mg to neutralize the system, 
# 729 in conc is 172 mM (deviating from 1 mM in the experiment, but we want to neutralize the system,
gmx insert-molecules -f test_box.gro -ci ../solutes/ADP.pdb -radius 2 -o test_box_1.gro -nmol 300
gmx insert-molecules -f test_box_1.gro -ci ../solutes/EDDA_MDA.pdb -radius 2 -o test_box_2.gro -nmol 425
gmx insert-molecules -f test_box_2.gro -ci ../solutes/EDDA_DDA.pdb -radius 2 -o test_box_3.gro -nmol 425
gmx insert-molecules -f test_box_3.gro -ci ../solutes/ACE.pdb -radius 2 -o test_box_4.gro -nmol 1700


# change the top
gmx solvate -cp test_box_4.gro -cs spc216.gro -o test_solvated.gro -p topol.top

# add water top path
# add salt ion top path

gmx grompp -f ions.mdp -c test_solvated.gro -p topol.top -o ions.tpr -maxwarn 0
gmx genion -s ions.tpr -o test_ions.gro -p topol.top -pname MG -pq 2 -np 729




gmx grompp -f em.mdp -c test_ions.gro -p topol.top -o em.tpr
gmx mdrun -deffnm em -v -ntmpi 1 -ntomp 24

# no need to constrain ligand pos?
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
gmx mdrun -deffnm nvt -v -ntmpi 1 -ntomp 12 -gpu_id 0

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -deffnm npt -v -ntmpi 1 -ntomp 12 -gpu_id 0

gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o test_md.tpr
nohup gmx mdrun -deffnm test_md -v -ntmpi 1 -ntomp 12 -gpu_id 0 > log.log &


gmx mdrun -deffnm npt -v -ntmpi 1 -ntomp 12 -gpu_id 0 -cpi npt.cpt

nohup gmx mdrun -deffnm test_md -v -ntmpi 1 -ntomp 12 -gpu_id 0 -cpi test_md.cpt > log.log &


echo "cd /home/alan/gromacs/groel_new/0212_ADP_MDA && /usr/local/gromacs/bin/gmx mdrun -deffnm test_md -v -ntmpi 1 -ntomp 24 -gpu_id 0 > log.log 2>&1" | at 05:00 2026-02-12



gmx trjconv -f test_md.xtc -s test_md.tpr -o out_cluster.xtc -pbc cluster -center -skip 10 -n index.ndx
gmx trjconv -f out_cluster.xtc -s test_md.tpr -o out_fit.xtc -fit rot+trans -center -n index.ndx
gmx trjconv -f out_fit.xtc -s test_md.tpr -o sk10_partial.pdb -center -n index.ndx


gmx make_ndx -f test_md.gro -o index.ndx 
gmx make_ndx -f test_md.gro -o index.ndx -n index.ndx
gmx trjconv -s test_md.tpr -f out_fit.xtc -o ref0.gro -n index.ndx -dump 0


gmx trjconv -f test_md.xtc -s test_md.tpr -o out_1_whole_ADP_sk1.xtc -pbc whole -center -skip 1 -n index.ndx 
gmx trjconv -s test_md.tpr -f out_1_whole_ADP_sk1.xtc -o out_1_whole_ADP_sk1_ref0.pdb -n index.ndx -dump 0


gmx trjconv -f test_md.xtc -s test_md.tpr -o out_1_whole_water_sk1.xtc -pbc whole -center -skip 1 -n index.ndx
gmx trjconv -s test_md.tpr -f out_1_whole_water_sk1.xtc -o out_1_whole_water_sk1_ref0.pdb -n index.ndx -dump 0



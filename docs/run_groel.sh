#!/usr/bin/env bash

# SMILES OF 3 forms of EDA
# NCCN 
# [NH3+]CCN 
# [NH3+]CC[NH3+]

# 1 uM Protein vs 200 mM EDDA,
# a box of   17.83700  10.68700  17.56300 nm is volume of 3348 nm^3
# 200 counts of EDDA is 200/(6.022e23 molecules/mol)/(3348e-24 L) = 0.0992 mol/L = 99.2 mM

# charge of groel is -133, 19 per subunit
# adding EDDA results in +200*2 + 200*1= +600 charge
# so need additional -467 charge from Ac- ions to neutralize system
# for NH4Ac system, add 400 NH4+ and 267 Ac- ions

echo -e "8\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n" | gmx pdb2gmx -f groel_2EU1.pdb -o test.gro -ignh -ter

gmx editconf -f test.gro -o test_box.gro -c -d 1.5

# for EDDA solute simulation
gmx insert-molecules -f test_box.gro -ci ../solutes/EDDA_MDA.pdb -radius 2 -o test_solutes_1_mol200.gro -nmol 200
gmx insert-molecules -f test_solutes_1_mol200.gro -ci ../solutes/EDDA_DDA.pdb -radius 2 -o test_solutes_2_mol200.gro -nmol 200
gmx insert-molecules -f test_solutes_2_mol200.gro -ci ../solutes/ACE.pdb -radius 2 -o test_solutes_3_mol200.gro -nmol 467

gmx solvate -cp test_solutes_3_mol200.gro -cs spc216.gro -o test_solvated.gro -p topol.top

# for AmAc solute simulation
gmx insert-molecules -f test_box.gro -ci ../solutes/NH4.pdb -radius 2 -o test_solutes_1_mol200.gro -nmol 400
gmx insert-molecules -f test_solutes_1_mol200.gro -ci ../solutes/ACE.pdb -radius 2 -o test_solutes_2_mol200.gro -nmol 267

gmx solvate -cp test_solutes_2_mol200.gro -cs spc216.gro -o test_solvated.gro -p topol.top

# gmx insert-molecules -f test_box.gro -ci ../solutes/EDDA_NDA.pdb -radius 2 -o test_solutes_.gro -conc 0.2



# TEST stucks here
# to solve the error "No default Proper Dih. types"
# go to the ffbonded.itp, and found if there is missing lines compared to the NDA.itp
# some lines might be missing and some might be duplicated, only copy the missing lines to the ffbonded.itp
gmx grompp -f ions.mdp -c test_solvated.gro -p topol.top -o ions.tpr -maxwarn 0
gmx genion -s ions.tpr -o test_ions.gro -p topol.top -pname NA -nname CL -neutral

gmx grompp -f em.mdp -c test_ions.gro -p topol.top -o em.tpr
gmx mdrun -deffnm em -v -ntmpi 1 -ntomp 24

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt -v -ntmpi 1 -ntomp 12 -gpu_id 1

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -deffnm npt -v -ntmpi 1 -ntomp 12 -gpu_id 1

gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o test_md.tpr
gmx mdrun -deffnm test_md -v -ntmpi 1 -ntomp 12 -gpu_id 0


# continue a job
gmx mdrun -s test_md.tpr -cpi test_md.cpt -deffnm test_md -v -ntmpi 1 -ntomp 12 -gpu_id 0


gmx trjconv -f test_md.xtc -s test_md.tpr -o test_out.pdb -fit rot+trans -center -skip 1000 

# make index file, to not include ACE residues
gmx make_ndx -f npt.gro -o index.ndx -n index.ndx
1 & ! r ACE
18 | 13 | 14

gmx trjconv -f test_md.xtc -s test_md.tpr -o cluster.xtc -pbc cluster -center -skip 1000 -n index.ndx
gmx trjconv -f cluster.xtc -s test_md.tpr -o cluster_fit.xtc -fit rot+trans -center -n index.ndx
gmx trjconv -f cluster_fit.xtc -s test_md.tpr -o sk1.pdb -center  -n index.ndx


# tried 18 18 19
# tried 18 19 19
# tried 19 19 19 break pro
# tried 19 18 19 break pro
gmx trjconv -f test_md.xtc -s test_md.tpr -o cluster35_DA.xtc -pbc cluster -center -skip 1000 -n index.ndx -e 40000 # 40ns
gmx trjconv -f cluster_DA.xtc -s test_md.tpr -o cluster_DA_fit.xtc -fit rot+trans -center -n index.ndx
gmx trjconv -f cluster_DA_fit.xtc -s test_md.tpr -o sk1_DA.pdb -center -n index.ndx



echo -e "19\n" | gmx trjconv \
  -s test_md.tpr -f test_md.xtc \
  -o skip1000_step1_whole.xtc \
  -pbc whole -skip 1000 \
  -n index.ndx

echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step1_whole.xtc \
  -o skip1000_step2_cluster.xtc \
  -pbc cluster -ur rect \
  -n index.ndx

echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step2_cluster.xtc \
  -o skip1000_step3_center.xtc \
  -center -pbc mol -ur rect \
  -n index.ndx


echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step3_center.xtc \
  -o skip1000_step4_fit.xtc \
  -fit rot+trans \
  -n index.ndx

echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step4_fit.xtc \
  -o skip1000_step5_fit_wrapped.pdb \
  -n index.ndx




awk '/^MODEL/ && ++n > 41 {exit} {if(n==41) print}' skip1000_step4_mol.pdb > model41.pdb


# re-do from 0
gmx trjconv \
  -s test_md.tpr -f test_md.xtc \
  -o skip1000_step1_cluster.xtc \
  -pbc cluster -skip 1000 \
  -n index.ndx

gmx trjconv \
  -s test_md.tpr -f skip1000_step1_cluster.xtc \
  -o skip1000_step2_center.pdb \
  -pbc mol -center \
  -n index.ndx

# they say use mol, nojump then it will be fixed
gmx trjconv \
  -s test_md.tpr -f skip1000_step2_center.pdb \
  -o skip1000_step3_mol.pdb \
  -center -pbc mol -ur rect \
  -n index.ndx

gmx trjconv \
  -s test_md.tpr -f skip1000_step3_mol.pdb \
  -o skip1000_step4_fit.pdb \
  -fit rot+trans -center -ur rect \
  -n index.ndx

##############################
# fit then pbc
gmx trjconv \
  -s test_md.tpr -f test_md.xtc \
  -o skip1000_step1_fit.xtc \
  -fit rot+trans -center -ur rect -skip 1000 \
  -n index.ndx

gmx trjconv \
  -s test_md.tpr -f skip1000_step1_fit.xtc \
  -o step2_cluster.xtc \
  -pbc cluster \
  -n index.ndx


remove solvent or inorganic or MPD
create 4ki8_A_obj, 4ki8 and chain A
create 1xck_A_obj, 1xck and chain A
align 4ki8_A_obj and polymer.protein, subunit_A_obj and polymer.protein
align 1xck_A_obj and polymer.protein, subunit_A_obj and polymer.protein
create ADP_obj, 4ki8_A_obj and resn ADP

# ===== Subunits =====
create subunit_A_obj, model model101 and resi 1-548
create subunit_B_obj, model model101 and resi 549-1096
create subunit_C_obj, model model101 and resi 1097-1644
create subunit_D_obj, model model101 and resi 1645-2192
create subunit_E_obj, model model101 and resi 2193-2740
create subunit_F_obj, model model101 and resi 2741-3288
create subunit_G_obj, model model101 and resi 3289-3836


# ===== Key residues (offset +548 each chain) =====
select key_A, subunit_A_obj and resi 52+53+87+89+90+398+495
select key_B, subunit_B_obj and resi 600+601+635+637+638+946+1043
select key_C, subunit_C_obj and resi 1148+1149+1183+1185+1186+1494+1591
select key_D, subunit_D_obj and resi 1696+1697+1731+1733+1734+2042+2139
select key_E, subunit_E_obj and resi 2244+2245+2279+2281+2282+2590+2687
select key_F, subunit_F_obj and resi 2792+2793+2827+2829+2830+3138+3235
select key_G, subunit_G_obj and resi 3340+3341+3375+3377+3378+3686+3783


# ===== Show sidechains =====
show sticks, (key_A or key_B or key_C or key_D or key_E or key_F or key_G) and sidechain


# ===== Per-chain interactive solutes (MDA first) =====
create MDA_A_obj, byres ((key_A around 5) and resname MDA)
create MDA_B_obj, byres ((key_B around 5) and resname MDA)
create MDA_C_obj, byres ((key_C around 5) and resname MDA)
create MDA_D_obj, byres ((key_D around 5) and resname MDA)
create MDA_E_obj, byres ((key_E around 5) and resname MDA)
create MDA_F_obj, byres ((key_F around 5) and resname MDA)
create MDA_G_obj, byres ((key_G around 5) and resname MDA)


# ===== Then DDA =====
create DDA_A_obj, byres ((key_A around 5) and resname DDA)
create DDA_B_obj, byres ((key_B around 5) and resname DDA)
create DDA_C_obj, byres ((key_C around 5) and resname DDA)
create DDA_D_obj, byres ((key_D around 5) and resname DDA)
create DDA_E_obj, byres ((key_E around 5) and resname DDA)
create DDA_F_obj, byres ((key_F around 5) and resname DDA)
create DDA_G_obj, byres ((key_G around 5) and resname DDA)

delete 4ki8 1xck 500000000


# select key_residues, resi 52+53+87+89+90+398
# select key_residues, \
# (resi 52+53+87+89+90+398) or \
# (resi 600+601+635+637+638+946) or \
# (resi 1148+1149+1183+1185+1186+1494) or \
# (resi 1696+1697+1731+1733+1734+2042) or \
# (resi 2244+2245+2279+2281+2282+2590) or \
# (resi 2792+2793+2827+2829+2830+3138) or \
# (resi 3340+3341+3375+3377+3378+3686)
# select interactive_solutes, byres ((key_residues around 5) and (resname MDA or resname DDA)) 

# select interactive_solutes_DDA, byres ((key_residues around 5) and (resname DDA)) 
# select subunit_F, resi 2741-3288


gmx sasa -f cluster_fit.xtc -s test_md.tpr -o sasa.xvg -n index.ndx
xmgrace sasa.xvg

gmx rms -f cluster_fit.xtc -s test_md.tpr -o rmsd.xvg -n index.ndx
xmgrace rmsd.xvg

xmgrace \
  ./0105_EDDA/sasa.xvg \
  ./0105_AMAC/sasa.xvg \
  -settype xy \
  -pexec "with g0; \
          legend on; \
          legend loctype view; \
          legend 0.78, 0.80; \
          s0 legend \"EDDA\"; \
          s1 legend \"AmAc\"; \
          s0 line color 2; \
          s1 line color 4"

xmgrace ./0105_EDDA/sasa.xvg \
  -settype xy \
  -pexec "with g0; \
          legend on; \
          s0 legend \"EDDA\"; \
          s0 line color 2; \
          s0 line linewidth 2"

xmgrace ./0105_AMAC/sasa.xvg \
  -settype xy \
  -pexec "with g0; \
          legend on; \
          s0 legend \"AmAc\"; \
          s0 line color 4; \
          s0 line linewidth 2"





# step 1 is to wrap EVERYTHING into box
echo -e "18\n" | gmx trjconv \
  -s test_md.tpr -f test_md.xtc \
  -o skip1000_step1_wrap.xtc \
  -pbc mol -ur rect \
  -skip 1 \
  -n index.ndx


# step 2 is to cluster and center again
echo -e "17\n17\n18\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step1_wrap.xtc \
  -o skip1000_step2_center.xtc \
  -center -pbc cluster -ur rect \
  -n index.ndx


# step 3a
echo -e "17\n18\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step2_center.xtc \
  -o skip1000_step3_fit.xtc \
  -fit rot+trans \
  -n index.ndx

# step 3b
# rewrap back into box after fitting
echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step3_fit.xtc \
  -o skip1000_step3_fit_wrapped1.xtc \
  -center -pbc mol -ur rect\
  -n index.ndx 


echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f skip1000_step3_fit_wrapped1.xtc \
  -o everything_fixed.pdb \
  -center -pbc mol -ur rect\
  -n index.ndx -skip 1000


scp -3 -r alan@128.194.145.182:/home/alan/gromacs/groel_new/solutes/ ./
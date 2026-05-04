#!/usr/bin/env bash

# i have 18 as my protein, while 19 as my whole system
# but the solutes in 19 always fly out of box, i fixed in step 4, but it "crashed" with overlapping atoms


'''
the first i tryied and i failed
'''
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
  -o skip1000_step4_fit_wrapped1.xtc \
  -center -pbc mol -ur rect\
  -n index.ndx 



'''
the 2nd i tryied and i failed
'''

# make protein continuous across PBC
echo -e "19\n" | gmx trjconv \
  -s test_md.tpr -f test_md.xtc \
  -o step0_nojump.xtc \
  -pbc nojump -ur rect \
  -n index.ndx -skip 1000

# center on protein, keep whole molecules
echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f step0_nojump.xtc \
  -o step1_center_mol.xtc \
  -center -pbc mol -ur compact \2
  -n index.ndx

# fit to protein, output whole system
echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f step1_center_mol.xtc \
  -o step2_fit_system.xtc \
  -fit rot+trans \
  -n index.ndx

# re-wrap after fit (optional)
echo -e "18\n19\n" | gmx trjconv \
  -s test_md.tpr -f step2_fit_system.xtc \
  -o final_fit_center_wrap.pdb \
  -center -pbc mol -ur compact \
  -n index.ndx


awk '/^MODEL/ && ++n > 101 {exit} {if(n==101) print}' final_fit_center_wrap.pdb > model101.pdb




nohup \
echo -e "17\n17\n0\n" | gmx trjconv \
  -s test_md.tpr -f test_md.xtc \
  -o st1_cluster_sk1.xtc \
  -center -pbc cluster -ur rect \
  -n index.ndx -skip 1 > log_cluster_sk1.txt 2>&1 &

echo -e "0\n" | gmx trjconv \
  -s test_md.tpr -f st1_cluster_sk1.xtc \
  -o st2_pbc.xtc \
  -pbc mol -ur rect \
  -n index.ndx -b 500000

gmx trjconv \
  -s test_md.tpr -f st2_pbc.xtc \
  -o st3_fit.xtc \
  -fit rot+trans -center -ur rect \
  -n index.ndx

gmx trjconv \
  -s test_md.tpr -f st2_pbc.xtc \
  -o test_sk100.xtc \
  -fit rot+trans -center -ur rect \
  -n index.ndx -skip 100

gmx trjconv \
  -s test_md.tpr -f st2_pbc.xtc \
  -o test_sk100.pdb \
  -fit rot+trans -center -ur rect \
  -n index.ndx -dump 500000
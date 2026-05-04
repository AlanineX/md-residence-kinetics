from pymol import cmd


# sele resi 452+461+463+464
# print(cmd.get_fastastr("obj01"))

# --- user settings ---
OBJ = "obj01"                        # object name as loaded in PyMOL
CHAINS = ["A", "B", "C", "D", "E", "F", "G"]  # chains to apply mutation
RESIDUES = [452, 461, 463, 464]     # positions to mutate
TARGET_AA = "ALA"                   # target mutation
ROTAMER_FRAME = "1"                 # rotamer frame to use
# ---------------------

def mutate_one(obj, chain, resi, target):
    """Perform mutation using the PyMOL Mutagenesis Wizard"""
    sel_path = f"/{obj}//{chain}/{resi}"
    cmd.wizard("mutagenesis")
    cmd.do("refresh_wizard")
    wiz = cmd.get_wizard()
    wiz.set_mode(target)
    wiz.do_select(sel_path)
    # choose the most probable rotamer
    cmd.frame(ROTAMER_FRAME)
    wiz.apply()
    cmd.set_wizard()  # exit wizard

# visualize original side chains
for c in CHAINS:
    cmd.select(f"to_mut_{c}", f"{OBJ} and chain {c} and resi {'+'.join(map(str, RESIDUES))}")
    cmd.show("sticks", f"to_mut_{c} and not name N+CA+C+O")
    cmd.color("yellow", f"to_mut_{c}")

# apply the mutations
for c in CHAINS:
    for r in RESIDUES:
        mutate_one(OBJ, c, r, TARGET_AA)
        print(f"Mutated chain {c} residue {r} to {TARGET_AA}")

# show mutated side chains
for c in CHAINS:
    cmd.show("sticks", f"{OBJ} and chain {c} and resi {'+'.join(map(str, RESIDUES))}")
    cmd.color("orange", f"{OBJ} and chain {c} and resi {'+'.join(map(str, RESIDUES))}")
    print(f"Colored chain {c} residue {r} to {TARGET_AA}")


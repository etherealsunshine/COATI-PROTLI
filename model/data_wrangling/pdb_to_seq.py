import os
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
parser = PDBParser(PERMISSIVE=1)
protein_pdb=[]
#goes over all the files and just gets the protein pdb
def pywalker(start_path):
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if "protein.pdb" in file:
                # Construct the full path to the file
                full_file_path = os.path.join(root, file)
                print(full_file_path)
                protein_pdb.append(full_file_path)
pywalker('data/P-L')
protein_data={}
def pdb_to_seq():
    for file in protein_pdb:
        dirname, structure_id = os.path.split(file)
        basename, extension = os.path.splitext(file)
        parts = basename.split('_')
        complex_id = parts[0]
        filename = file
        structure = parser.get_structure(structure_id, filename)
        # 2. Use Polypeptide Builder
        ppb = PPBuilder()
        all_chains=[]
        for pp in ppb.build_peptides(structure):
            all_chains.append((pp.get_sequence(),len(pp)))
        max_length = max(all_chains,key=lambda x: x[1])[0]
        protein_data[complex_id] = max_length

pdb_to_seq()



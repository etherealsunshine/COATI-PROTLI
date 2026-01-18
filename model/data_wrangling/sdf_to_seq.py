import os
from rdkit import Chem
# go over all the files and get the sdf
ligand_sdf=[]
def pywalker(start_path):
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if "ligand.sdf" in file:
                # Construct the full path to the file
                full_file_path = os.path.join(root, file)
                print(full_file_path)
                ligand_sdf.append(full_file_path)
pywalker('data/P-L')
smiles_dict={}
def sdf_to_smiles():
    for file in ligand_sdf:
        dirname, structure_id = os.path.split(file)
        basename, extension = os.path.splitext(file)
        parts = basename.split('_')
        complex_id = parts[0]

        sdf_supplier = Chem.SDMolSupplier(file)
        all_smiles=[]
        for mol in sdf_supplier:
            if mol is not None:
                # Get the SMILES string (canonical by default)
                smi = Chem.MolToSmiles(mol)
                all_smiles.append(smi)
        smiles_dict[complex_id] = all_smiles[0]

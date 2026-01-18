import pandas as pd
import pdb_to_seq
import sdf_to_seq

protein_data = pdb_to_seq.pdb_to_seq()
ligand_data  = sdf_to_seq.sdf_to_smiles()


def build_dataset(protein_data=protein_data,ligand_data=ligand_data):
    common_keys = protein_data.keys() & ligand_data.keys()
    rows=[]
    for cid in common_keys:
        rows.append({
            "complex_id":cid,
            "protein":protein_data[cid],
            "ligand_smiles":ligand_data[cid]
        })
    df = pd.DataFrame(rows)
    return df

df = build_dataset()
print(df)
print(len(protein_data), len(ligand_data))
df.to_csv('protein_ligand_data.csv')  

import torch
from torch.utils.data import Dataset
import pandas as pd

class ProteinLigandDataset(Dataset):
    def __init__(self,data_path):
        self.data = pd.read_csv(data_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            "complex_id":row["complex_id"],
            "protein":row["protein"],
            "ligand_smiles":row["ligand_smiles"]
        }

def collate_protein_ligand(batch,protein_tokenizer,ligand_tokenizer):
    proteins = [b["protein"] for b in batch]
    ligands = [b["ligand_smiles"] for b in batch]
    ids = [b["complex_id"] for b in batch]

    protein_tokens = protein_tokenizer(
        proteins,
        padding = True,
        truncation = True,
        return_tensors = "pt"
    )

    ligand_tokens = ligand_tokenizer(
        ligands,
        padding=True,
        truncation = True,
        return_tensors = "pt"
    )

    return {
        "protein": protein_tokens,
        "ligand" : ligand_tokens,
        "id":ids
    }
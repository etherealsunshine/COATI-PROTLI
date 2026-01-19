import torch
from torch.utils.data import Dataset
import pandas as pd

#from model.precompute_ligand_embeddings import ligand_embeddings

class ProteinLigandDataset(Dataset):
    def __init__(self,data_path,ligand_emb_path):
        self.data = pd.read_csv(data_path)
        self.ligand_embs = torch.load(ligand_emb_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            "complex_id":row["complex_id"],
            "protein":row["protein"],
            #"ligand_smiles":row["ligand_smiles"]
            "ligand_embedding":self.ligand_embs[row["complex_id"]]
        }

def collate_protein_ligand(batch,protein_tokenizer):
    proteins = [b["protein"] for b in batch]
    #ligands = [b["ligand_smiles"] for b in batch]
    ligand_embeddings = torch.stack([b["ligand_embedding"] for b in batch])
    ids = [b["complex_id"] for b in batch]

    protein_tokens = protein_tokenizer(
        proteins,
        padding = True,
        truncation = True,
        return_tensors = "pt"
    )

    """ligand_tokens = ligand_tokenizer(
        ligands,
        padding=True,
        truncation = True,
        return_tensors = "pt"
    )"""

    return {
        "protein": protein_tokens,
        "ligand_embeddings":ligand_embeddings,
        "id":ids
    }
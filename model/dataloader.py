import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
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


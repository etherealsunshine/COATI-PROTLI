import torch
import torch.nn as nn
from coati.models.simple_coati2.io import load_coati2
from coati.generative.coati_purifications import embed_smiles_batch
from rdkit import Chem

# Model parameters are pulled from the url and stored in a local models/ dir.
encoder, tokenizer = load_coati2(
    freeze=True,
    device=torch.device("cpu"),
    # model parameters to load.
    doc_url="s3://terray-public/models/coati2_chiral_03-08-24.pkl",
)

class LigandEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,smiles_batch):
        canon_smiles = [Chem.CanonSmiles(smile) for smile in smiles_batch]
        vecs = embed_smiles_batch(canon_smiles, encoder, tokenizer)
        return vecs

lig_enc = LigandEncoder()
out = lig_enc(["c1ccccc1", "CCO"])
print(out.shape)


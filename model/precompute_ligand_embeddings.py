#this file eists because coati's RDKIT dependency needs <3.12 python to function. 
# Precompute the embeddings since we dont finetune the encoder anyways.
# Might downgrade overall project in the future to accomodate for this

import pandas as pd
import torch
from rdkit import Chem
from coati.models.simple_coati2.io import load_coati2
from coati.generative.coati_purifications import embed_smiles_batch

df = pd.read_csv("/Users/utkarsh/COATI-PROTLI/protein_ligand_data.csv")

# load COATI
encoder, tokenizer = load_coati2(
    freeze=True,
    doc_url="s3://terray-public/models/coati2_chiral_03-08-24.pkl",
)

ligand_embeddings = {}

BATCH_SIZE = 32 

for i in range(0,len(df),BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]

    cids = batch["complex_id"].tolist()
    smiles = batch["ligand_smiles"].tolist()

    smiles = [Chem.CanonSmiles(s) for s in smiles]
    with torch.no_grad():
        vecs = embed_smiles_batch(smiles, encoder, tokenizer)

    for cid, vecs in zip(cids,vecs):
        ligand_embeddings[cid]=vecs.cpu()

torch.save(ligand_embeddings, "ligand_embeddings.pt")
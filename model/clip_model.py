import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinLigandClip(nn.Module):
    def __init__(self,protein_encoder,ligand_encoder,protein_proj,ligand_proj,temperature=0.07):
        super().__init__()
        self.protein_encoder = protein_encoder
        self.ligand_encoder =ligand_encoder
        self.protein_proj = protein_proj
        self.ligand_proj = ligand_proj

        self.logit_scale = nn.Parameter(
            torch.tensor(1/temperature).log()
        )
    
    def forward(self,protein_inputs,ligand_smiles):
        #encode inputs

        protein_embeddings = self.protein_encoder(
            input_ids = protein_inputs["input_ids"],
            attention_mask = protein_inputs["attention_mask"]
        )

        ligand_embeddings = self.ligand_encoder(ligand_smiles)

        #projection
        protein_proj = self.protein_proj(protein_embeddings)
        ligand_proj = self.ligand_proj(ligand_embeddings)

        #normalziaiton
        protein_proj = F.normalize(protein_proj,dim=-1)
        ligand_proj = F.normalize(ligand_proj,dim=-1)

        return protein_proj,ligand_proj

def clip_loss(protein_proj,ligand_proj,logit_scale):
    logits = logit_scale.exp() * protein_proj @ ligand_proj.T
    labels = torch.arange(len(protein_proj),device=protein_proj.device)
    loss_p2l = F.cross_entropy(logits,labels)
    loss_l2p = F.cross_entropy(logits.T,labels)
    return (loss_p2l+loss_l2p)/2
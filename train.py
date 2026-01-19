import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.dataloader import ProteinLigandDataset, collate_protein_ligand
from model.encoders  import ProteinEncoder
from model.projection import ProjectionHead
from model.clip_model import ProteinLigandClip, clip_loss

# Config
DEVICE = "mps" if torch.mps.is_available() else "cpu"
BATCH_SIZE = 4
LR = 1e-3

CSV_PATH = "protein_ligand_data.csv"
LIGAND_EMB_PATH = "ligand_embeddings.pt"


# Tokenizer
protein_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/esm2_t33_650M_UR50D"
)


dataset = ProteinLigandDataset(CSV_PATH, LIGAND_EMB_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_protein_ligand(b, protein_tokenizer),
)

protein_encoder = ProteinEncoder().to(DEVICE)

protein_proj = ProjectionHead(1280,128).to(DEVICE)
ligand_proj = ProjectionHead(512,128).to(DEVICE)

model = ProteinLigandClip(
    protein_encoder=protein_encoder,
    #ligand_encoder
    protein_proj=protein_proj,
    ligand_proj=ligand_proj,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(protein_proj.parameters()) +
    list(ligand_proj.parameters()) +
    [model.logit_scale],
    lr=LR
)


model.train()

for step, batch in enumerate(loader):
    if step >= 3:
        break

    protein_inputs = {
        k: v.to(DEVICE) for k, v in batch["protein"].items()
    }
    ligand_embs = batch["ligand_embeddings"].to(DEVICE)

    optimizer.zero_grad()

    protein_proj_out, ligand_proj_out = model(
        protein_inputs=protein_inputs,
        ligand_embeddings=ligand_embs,  # naming is legacy
    )

    loss = clip_loss(
        protein_proj_out,
        ligand_proj_out,
        model.logit_scale,
    )

    loss.backward()
    optimizer.step()

    print(f"Step {step} | Loss: {loss.item():.4f}")

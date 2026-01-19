import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.dataloader import ProteinLigandDataset, collate_protein_ligand
from model.encoders  import ProteinEncoder
from model.projection import ProjectionHead
from model.clip_model import ProteinLigandClip, clip_loss
from model.metrics import clip_metrics
from model.viz_similarity import plot_similarity

# Config
DEVICE = "mps" if torch.mps.is_available() else "cpu"
BATCH_SIZE = 16
LR = 1e-3
NUM_EPOCHS = 50  

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

protein_proj = ProjectionHead(1280, 128).to(DEVICE)
ligand_proj = ProjectionHead(512, 128).to(DEVICE)

model = ProteinLigandClip(
    protein_encoder=protein_encoder,
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


for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    epoch_top1 = 0.0
    epoch_top5 = 0.0
    num_batches = 0
    
    for step, batch in enumerate(loader):
        protein_inputs = {
            k: v.to(DEVICE) for k, v in batch["protein"].items()
        }
        ligand_embs = batch["ligand_embeddings"].to(DEVICE)

        optimizer.zero_grad()

        protein_proj_out, ligand_proj_out = model(
            protein_inputs=protein_inputs,
            ligand_embeddings=ligand_embs,
        )

        loss = clip_loss(
            protein_proj_out,
            ligand_proj_out,
            model.logit_scale,
        )

        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            metrics = clip_metrics(
                protein_proj_out,
                ligand_proj_out,
                model.logit_scale
            )
            
        epoch_loss += loss.item()
        epoch_top1 += metrics['top1']
        epoch_top5 += metrics['top5']
        num_batches += 1

        # Print every 10 steps
        if step % 10 == 0:
            logits = model.logit_scale.exp() * protein_proj_out @ ligand_proj_out.T
            idx = 0
            k = min(5, logits.size(1))
            topk = logits[idx].topk(k).indices.tolist()
            
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step} | "
                f"Loss: {loss.item():.4f} | "
                f"Top1: {metrics['top1']:.3f} | "
                f"Top5: {metrics['top5']:.3f}"
            )

        # Plot similarity every 10 epochs on first batch
        if step == 0 and (epoch % 10 == 0 or epoch == NUM_EPOCHS - 1):
            plot_similarity(
                protein_proj_out,
                ligand_proj_out,
                model.logit_scale,
                title=f"Protein–Ligand Similarity (Epoch {epoch+1})"
            )
    
    # Print epoch summary
    avg_loss = epoch_loss / num_batches
    avg_top1 = epoch_top1 / num_batches
    avg_top5 = epoch_top5 / num_batches
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
    print(f"Avg Loss: {avg_loss:.4f} | Avg Top1: {avg_top1:.3f} | Avg Top5: {avg_top5:.3f}")
    print(f"{'='*60}\n")

print("Training done")
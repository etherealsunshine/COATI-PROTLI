# modal/modal_train.py
import modal

# Define the Modal app
app = modal.App("coati-protli-training")

# Define the image with your dependencies AND data files
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "biopython>=1.86",
        "matplotlib>=3.10.8",
        "numpy>=2.4.1",
        "pandas>=2.3.3",
        "rdkit>=2025.9.3",
        "torch>=2.9.1",
        "torchvision>=0.24.1",
        "transformers",
        "seaborn",
    )
    # Add your model directory
    .add_local_dir("model", remote_path="/root/model")
    # Add your data files directly to the image
    .add_local_file("protein_ligand_data.csv", remote_path="/root/protein_ligand_data.csv")
    .add_local_file("ligand_embeddings.pt", remote_path="/root/ligand_embeddings.pt")
)

checkpoint_volume = modal.Volume.from_name("coati-checkpoints", create_if_missing=True)

@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/checkpoints": checkpoint_volume,
    },
    timeout=3600,
)
def train():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import sys
    
    sys.path.insert(0, "/root")
    
    from model.dataloader import ProteinLigandDataset, collate_protein_ligand
    from model.encoders import ProteinEncoder
    from model.projection import ProjectionHead
    from model.clip_model import ProteinLigandClip, clip_loss
    from model.metrics import clip_metrics
    
    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    BATCH_SIZE = 8
    LR = 1e-3
    NUM_EPOCHS = 5
    
    # Data files are now in /root
    CSV_PATH = "/root/protein_ligand_data.csv"
    LIGAND_EMB_PATH = "/root/ligand_embeddings.pt"
    
    print(f"Loading data from {CSV_PATH} and {LIGAND_EMB_PATH}...")
    
    protein_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t33_650M_UR50D"
    )
    
    dataset = ProteinLigandDataset(CSV_PATH, LIGAND_EMB_PATH)
    print(f"Dataset size: {len(dataset)}")
    
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
    
    print("Starting training...")
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
            
            if step % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Top1: {metrics['top1']:.3f} | "
                    f"Top5: {metrics['top5']:.3f}"
                )
        
        avg_loss = epoch_loss / num_batches
        avg_top1 = epoch_top1 / num_batches
        avg_top5 = epoch_top5 / num_batches
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"Avg Loss: {avg_loss:.4f} | Avg Top1: {avg_top1:.3f} | Avg Top5: {avg_top5:.3f}")
        print(f"{'='*60}\n")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f"/checkpoints/checkpoint_epoch_{epoch+1}.pt")
        checkpoint_volume.commit()
        print(f"Saved checkpoint at epoch {epoch+1}")
    
    torch.save(model.state_dict(), "/checkpoints/final_model.pt")
    checkpoint_volume.commit()
    print("Training complete!")
    
    return {
        "final_loss": avg_loss,
        "final_top1": avg_top1,
        "final_top5": avg_top5
    }


@app.local_entrypoint()
def main():
    print("Starting Modal training job...")
    result = train.remote()
    print(f"\nFinal Results: {result}")
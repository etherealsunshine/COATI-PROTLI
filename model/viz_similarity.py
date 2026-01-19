import matplotlib.pyplot as plt
import torch

def plot_similarity(protein_emb, ligand_emb, logit_scale, title="Similarity"):
    logits = logit_scale.exp() * protein_emb @ ligand_emb.T
    plt.imshow(logits.detach().cpu())
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Ligand index")
    plt.ylabel("Protein index")
    plt.show()

(Disclaimer: Used GenAI to write this)

# PROTLI — Protein–Ligand Contrastive Learning

**PROTLI** is a CLIP-style multimodal representation learning framework that aligns **protein sequences** and **small-molecule ligands** into a shared latent space using contrastive learning.

The model combines:

* **ESM2** (FAIR) for protein sequence embeddings
* **COATI** (Terray Therapeutics) for ligand (SMILES) embeddings
* Lightweight **projection heads** trained with an InfoNCE-style contrastive objective

The goal is to learn embeddings where cognate protein–ligand pairs are close, and non-matching pairs are far apart.

---

## Motivation

Protein–ligand modeling typically relies on:

* Docking (slow, structure-dependent)
* Supervised affinity prediction (label-limited)
* Separate encoders without joint alignment

This project explores a **representation-first alternative**:

> Can we learn a shared embedding space for proteins and ligands using only known binding pairs?

Such embeddings can support:

* Ligand retrieval for a target protein
* Protein-conditioned ligand filtering
* Initialization for downstream affinity or generative models

---

## Dataset

* Protein–ligand complexes derived from a PDBBind-style directory structure
* For each complex:

  * **Protein**: longest polypeptide chain extracted from PDB
  * **Ligand**: canonical SMILES from SDF

Ligand embeddings are **precomputed and cached** using COATI to simplify training and avoid environment conflicts.

---

## Architecture

### Protein Encoder

* **Model**: `facebook/esm2_t33_650M_UR50D`
* Frozen weights
* Mean pooling over residue embeddings with attention masking

### Ligand Encoder

* **Model**: COATI2 (Terray Therapeutics)
* Frozen weights
* Ligand embeddings precomputed and loaded from disk

### Projection Heads

* Linear → LayerNorm → L2 normalization
* Output dimension: 128
* Trainable

### Objective

* Symmetric CLIP-style contrastive loss (InfoNCE)
* Trainable temperature parameter

---

## Training Overview

Only the **projection heads** and **temperature** are trained.

```text
Protein sequence  → ESM2 → pooled embedding → projection
Ligand SMILES     → COATI → cached embedding → projection
                                ↓
                      Contrastive (CLIP) loss
```

---

## Repository Structure

```
.
├── model/
│   ├── dataloader.py              # Dataset + collate_fn
│   ├── encoders.py                # Protein encoder (ESM2)
│   ├── projection.py              # Projection head
│   ├── clip_model.py              # CLIP-style multimodal model + loss
│   └── precompute_ligand_embeddings.py
│
├── train.py                       # Minimal training loop
├── protein_ligand_data.csv        # Protein–ligand pairs
├── ligand_embeddings.pt           # Cached COATI embeddings
└── README.md
```

---

## Environment Notes

* Protein model (ESM2): PyTorch + HuggingFace Transformers
* Ligand model (COATI): separate Python environment required due to RDKit constraints

To avoid conflicts:

* Ligand embeddings are **precomputed once**
* Training uses a **single environment** thereafter

---

## Current Status

* [x] Dataset construction
* [x] Protein encoder (ESM2)
* [x] Ligand encoder (COATI, cached)
* [x] CLIP-style contrastive training
* [ ] Quantitative evaluation (retrieval metrics)
* [ ] Scaling to larger datasets
* [ ] Optional fine-tuning of encoders

---

## Future Directions

* Protein → ligand retrieval benchmarks
* Hard negative mining
* Joint fine-tuning of encoders
* Extension to structure-aware protein models
* Conditioning generative ligand models on protein embeddings

---

## Acknowledgements

* Meta AI — ESM2
* Terray Therapeutics — COATI
* HuggingFace Transformers
* PyTorch

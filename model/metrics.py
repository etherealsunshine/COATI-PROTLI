import torch
import torch.nn.functional as F

def clip_metrics(protein_emb, ligand_emb, logit_scale):
   
    logits = logit_scale.exp() * protein_emb @ ligand_emb.T
    labels = torch.arange(logits.size(0), device=logits.device)

    top1 = (logits.argmax(dim=1) == labels).float().mean()
    k = min(5, logits.size(1))
    
    top5 = (
        logits.topk(k, dim=1).indices == labels.unsqueeze(1)
    ).any(dim=1).float().mean()

    return {
        "top1": top1.item(),
        "top5": top5.item(),
        "loss_p2l": F.cross_entropy(logits, labels).item(),
        "loss_l2p": F.cross_entropy(logits.T, labels).item(),
    }

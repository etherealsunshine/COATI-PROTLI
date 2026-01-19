import torch 
from transformers import AutoModel,AutoTokenizer


#now we'll do mean pooling with masking for the encoder . Maybe try attention pooling for future?

def mean_pool(last_hidden_state,attention_mask):

    mask = attention_mask.unsqueeze(-1).float()
    masked_embbedings = last_hidden_state*mask

    summed = masked_embbedings.sum(dim=1)
    counts = mask.sum(dim=1)

    return summed/counts.clamp(min=1e-9)

#use 650M for now
class ProteinEncoder(torch.nn.Module):
    def __init__(self,model_name="facebook/esm2_t33_650M_UR50D",freeze=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self,input_ids,attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask
        )
        last_hidden = outputs.last_hidden_state
        pooled = mean_pool(last_hidden,attention_mask)
        return pooled




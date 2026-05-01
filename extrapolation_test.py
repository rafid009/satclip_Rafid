from huggingface_hub import hf_hub_download
from satclip.load import get_satclip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

c = torch.randn(32, 2)  # Represents a batch of 32 locations (lon/lat)

model = get_satclip(
    hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
    device=device,
)  # Only loads location encoder by default
model.eval()
with torch.no_grad():
    emb = model(c.double().to(device)).detach().cpu()
    print(emb.shape)  # Should print (32, 512) for the location embeddings
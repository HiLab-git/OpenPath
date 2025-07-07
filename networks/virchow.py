import timm
from timm.layers import SwiGLUPacked
import torch
from huggingface_hub import login

# login()

print(timm.__version__)
model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model.eval()

# image = torch.rand((1,3,224,224))
# output = model(image)  # size: 1 x 257 x 1280

# class_token = output[:, 0]    # size: 1 x 1280
# patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

# # concatenate class token and average pool of patch tokens
# embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
# print(embedding.shape)

ckpt = torch.load('/home/ubuntu/data/lanfz/checkpoints/Virchow/pytorch_model.bin',map_location="cpu")
model.load_state_dict(ckpt, strict=True)
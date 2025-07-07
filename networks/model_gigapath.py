import timm
from PIL import Image
from torchvision import transforms
import torch
import sys
sys.path.append('/home/ubuntu/data/lanfz/codes/openset-AL/')
from prov_gigapath.gigapath import slide_encoder
# from prov_gigapath.gigapath.pipeline import run_inference_with_tile_encoder



slide_encoder_giga = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
# print(list(slide_encoder_giga.children()))
with torch.cuda.amp.autocast(dtype=torch.float16):
    patch_embed = torch.rand((1, 100, 1536))
    coords = torch.rand((1, 100, 2))

output_slide_embed = slide_encoder_giga(patch_embed, coords)
# tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)

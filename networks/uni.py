import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class UNI_fc(nn.Module):
    def __init__(self, num_class=2, linear_pro=False) -> None:
        super(UNI_fc, self).__init__()
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        ckpt = torch.load('/home/ubuntu/data/lanfz/checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin',map_location="cpu")
        # print(ckpt)
        model.load_state_dict(ckpt, strict=True)
        # for name, module in model.named_modules():
        #     print(name)
        self.feature_extractor = model
        if linear_pro:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

def get_uni_fc_lora(num_class=2):
    model = UNI_fc(num_class)
    # target_modules = ["encoder.layers.{0}.self_attn.v_proj".format(i) for i in range(12)] + \
    #     ["encoder.layers.{0}.self_attn.q_proj".format(i) for i in range(12)]
    target_modules = ['attn.qkv']
    # target_modules = ["encoder.layers.{0}.self_attn.v_proj".format(i) for i in range(12)] + \
    #     ["encoder.layers.{0}.self_attn.q_proj".format(i) for i in range(12)] + \
    #     ["encoder.layers.{0}.self_attn.k_proj".format(i) for i in range(12)]
    # print(target_modules)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules, # ['k_proj', 'v_proj']
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["fc"],
        # modules_to_save=["fc"],
    )

    lora_model = get_peft_model(model, config)
    return lora_model

if __name__ == "__main__":
    # model = UNI_fc()
    model = get_uni_fc_lora()
    print_trainable_parameters(model)
    input_tensor = torch.rand((2,3,224,224))
    out = model(input_tensor)
    print(out.shape)
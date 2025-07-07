import transformers
import accelerate
import peft
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import timm
import lightning as L
from torch import optim, nn, utils, Tensor


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

class ViT(nn.Module):
    def __init__(self, num_class=2):
        super(ViT, self).__init__()
        model_checkpoint = "/home/ubuntu/data/lanfz/codes/CLIP-main/vit-base-in21k"

        model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        model.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))   
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out.logits

class ViT_tiny(nn.Module):
    def __init__(self, pretrained, num_class=2):
        super(ViT_tiny, self).__init__()
        model = timm.create_model("hf_hub:timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained)
        model.head = nn.Linear(192, num_class, bias=True)
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out

class ViT_small(nn.Module):
    def __init__(self, pretrained, num_class=2):
        super(ViT_small, self).__init__()
        model = timm.create_model("hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained)
        model.head = nn.Linear(384, num_class, bias=True)
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out

def get_vit_fea(num_class=2):
    model_checkpoint = "/home/ubuntu/data/lanfz/codes/CLIP-main/vit-base-in21k"

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )
    model.classifier = nn.Sequential(nn.Linear(768, num_class))
    # print(model.eval())
    # print(model.classifier.eval())
    # del model.classifier
    
    return model

def get_vit_lora(num_class=2):
    model = ViT(num_class=num_class)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    # print(model.eval())
    lora_model = get_peft_model(model, config)
    return lora_model

# class Vit_tiny_Lit(L.LightningModule):
#     def __init__(self, pretrain, num_class):
#         super().__init__()
#         model = timm.create_model("hf_hub:timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrain)
#         model.head = nn.Linear(192, num_class, bias=True)
#         self.model = model

#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         # it is independent of forward
#         x, y = batch['img'], batch['cls_label']
#         # print(x.shape)
#         logits = self.model(x)
#         loss = nn.functional.cross_entropy(logits, y)
#         acc = accuracy(logits, y, topk=(1,))
#         # Logging to TensorBoard (if installed) by default
#         self.log("train_loss", loss)
#         self.log('train_acc', acc[0])
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch['img'], batch['cls_label']
#         # print(x.shape)
#         logits = self.model(x)
#         loss = nn.functional.cross_entropy(logits, y)
#         acc = accuracy(logits, y, topk=(1,))
#         # Logging to TensorBoard (if installed) by default
#         self.log("train_loss", loss)
#         self.log('train_acc', acc[0])
#         return loss

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

if __name__ == "__main__":
    # model
    # model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")

    model = ViT_small()
    print(list(model.named_children()))
    image = torch.rand((2,3,224,224))
    # model = get_vit_fea()
    # model = get_vit_lora()
    print_trainable_parameters(model)
    out = model(image)
    print(out.shape)
    # out = model(image, output_hidden_states=True)
    # print(torch.mean(out.hidden_states[-1], dim=1).shape)
import lightning.pytorch as L
from open_clip import create_model_from_pretrained, get_tokenizer
import argparse
import torch
import os
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from torch import optim, nn, utils, Tensor
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torchvision.models as models
import torch.nn.functional as F
import timm
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config
from transformers import CLIPProcessor, CLIPModel
import copy
from torchmetrics import Accuracy, F1Score, Precision, Recall


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    # print(f"Test Accuracy: {test_accuracy.item()}, f1:{f1}, precision:{p}, recall:{r}")
    return test_accuracy, f1, r, p

def cls_count(args, y_pred):
    count = [0]*args.num_class
    for item in y_pred:
        count[item] += 1
    print(count)
    return count

class ResNet18_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        if pretrain:
            self.net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.net = models.resnet18()
        self.net.fc = nn.Linear(512, num_class)
        self.args = args
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        # optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class ResNet18_Lit2(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        if pretrain:
            self.net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.net = models.resnet18()
        self.net.fc = nn.Linear(512, num_class)
        self.args = args
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        # optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ResNet50_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        if pretrain:
            self.net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.net = models.resnet50()
        self.net.fc = nn.Linear(2048, num_class)
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.net(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class Vit_tiny_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model = timm.create_model("hf_hub:timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrain)
        model.head = nn.Linear(192, num_class, bias=True)
        self.model = model
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_acc', acc[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.model(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class BMC_LoRA_LIP_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['qkv'],
            lora_dropout=0.1,
            bias="none",
        )
        if num_class == 3:
            text_prompt = ["An H&E image of lymphocytes","An H&E image of normal mucosa", "An H&E image of adenocarcinoma epithelium"]
        lora_model = get_peft_model(BMC_model, config)
        self.model = lora_model
        self.texts = BMC_tokenizer(text_prompt).cuda()
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        y_pred = np.array(logits.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class BMC_LIP_FT_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        all_prompt = ["An H&E image of Adipose",'An H&E image of background',"An H&E image of debris",\
                   "An H&E image of lymphocytes", "An H&E image of mucus","An H&E image of smooth muscle",
                   "An H&E image of normal mucosa","An H&E image of cancer-associated stroma",
                   "An H&E image of adenocarcinoma epithelium"
                    ]
        text_prompt = np.array(all_prompt)[np.array(args.id_cls)]
        self.model = BMC_model
        self.texts = BMC_tokenizer(text_prompt).cuda()
        for name, params in self.model.named_parameters():
            if 'visual' not in name:
                params.requires_grad = False
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        y_pred = np.array(logits.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_LIP_Vision_FT_Lit(L.LightningModule):
    def __init__(self, pretrain, args):
        super().__init__()
        BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        all_prompt = ["An H&E image of Adipose",'An H&E image of background',"An H&E image of debris",\
                   "An H&E image of lymphocytes", "An H&E image of mucus","An H&E image of smooth muscle",
                   "An H&E image of normal mucosa","An H&E image of cancer-associated stroma",
                   "An H&E image of adenocarcinoma epithelium"
                    ]
        text_prompt = np.array(all_prompt)[np.array(args.id_cls)]
        self.model = BMC_model
        self.texts = BMC_tokenizer(text_prompt).cuda()
        # frozen text encoder
        for name, param in self.model.named_parameters():
            if 'visual' not in name:
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, args.num_class))
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t()) # LIP outputs

        logits_fc = self.fc(self.model.visual(x, hidden=True))

        loss = (nn.functional.cross_entropy(logits, y) + nn.functional.cross_entropy(logits_fc, y))/2
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())
        logits_fc = self.fc(self.model.visual(x, hidden=True))
        logits = (logits.softmax(1) + logits_fc.softmax(1))/2

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        y_pred = np.array(logits.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_LIP_VLP_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        if num_class == 3:
            text_prompt = ["An H&E image of lymphocytes","An H&E image of normal mucosa", "An H&E image of adenocarcinoma epithelium"]
        self.model = BMC_model
        self.texts = BMC_tokenizer(text_prompt).cuda()
        for name, params in self.model.named_parameters():
            if 'visual' not in name:
                params.requires_grad = False
        self.VLP = nn.Linear(512, 512, bias=False)
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        image_features_new = self.VLP(image_features)
        logits = (logit_scale * image_features_new @ text_features.t())

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, text_features, logit_scale = self.model(x, self.texts)
        logits = (logit_scale * image_features @ text_features.t())

        y_pred = np.array(logits.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_Vision_FT_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.feature_extractor = model.visual
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args
        self.accuracy = Accuracy(task="multiclass", num_classes=num_class)
        self.f1 = F1Score(task="multiclass", num_classes=num_class)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        acc, f1, r, p = cal_acc(y_pred=x.argmax(1).cpu().numpy(), y_true=y.cpu().numpy())
        # Logging to TensorBoard (if installed) by default
        self.accuracy.update(x, y)
        self.f1.update(x, y)
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', r, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.accuracy.update(x, y)
        self.f1.update(x, y)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def on_validation_epoch_end(self):
        val_acc = self.accuracy.compute()
        val_f1 = self.f1.compute()
        self.log("val_acc_epoch", val_acc, prog_bar=True)
        self.log("val_f1_epoch", val_f1, prog_bar=True)
        self.accuracy.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        test_acc = self.accuracy.compute()
        test_f1 = self.f1.compute()
        self.log("test_acc_epoch", test_acc, prog_bar=True)
        self.log("test_f1_epoch", test_f1, prog_bar=True)
        self.accuracy.reset()
        self.f1.reset()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_Vision_FT_Detec_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.feature_extractor = model.visual
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        x = x / 0.5
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_Vision_LoRA_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.feature_extractor = model.visual
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['qkv'],
            lora_dropout=0.1,
            bias="none",
        )
        self.feature_extractor = get_peft_model(self.feature_extractor, config)
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_Vision_FT_Lit_Label_Change(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.feature_extractor = model.visual
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args
        self.pseudo_records = []
        self.cur_batch_pseudo = []
        self.pseudo_names = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['img']
        # record pseudo labels
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        x = list(x.argmax(1).cpu().numpy())
        if batch_idx == 0:
            self.cur_batch_pseudo = []
            self.pseudo_names = []
        self.cur_batch_pseudo += x
        self.pseudo_names += batch['img_name']
        # print('----------', self.trainer.num_val_batches[0])
        if batch_idx == self.trainer.num_val_batches[0] - 1:
            self.pseudo_records.append(self.cur_batch_pseudo)   
        return None

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class PLIP_LIP_FT_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        PLIP_model = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
        PLIP_processor = CLIPProcessor.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
        if num_class == 3:
            text_prompt = ["An H&E image of lymphocytes","An H&E image of normal mucosa", "An H&E image of adenocarcinoma epithelium"]
        self.inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
        self.model = PLIP_model
        for params in self.model.text_model.parameters():
            params.requires_grad = False
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        self.inputs['pixel_values'] = x

        for key in self.inputs.keys():
            self.inputs[key] = self.inputs[key].cuda()
        outputs = self.model.forward(**self.inputs)
        # this is the image-text similarity score
        # print('PLIP logit scale:', PLIP_model.logit_scale)
        logits = outputs.logits_per_image

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        self.inputs['pixel_values'] = x

        for key in self.inputs.keys():
            self.inputs[key] = self.inputs[key].cuda()
        outputs = self.model.forward(**self.inputs)
        # this is the image-text similarity score
        # print('PLIP logit scale:', PLIP_model.logit_scale)
        logits = outputs.logits_per_image

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        self.inputs['pixel_values'] = x

        for key in self.inputs.keys():
            self.inputs[key] = self.inputs[key].cuda()
        outputs = self.model.forward(**self.inputs)
        # this is the image-text similarity score
        # print('PLIP logit scale:', PLIP_model.logit_scale)
        logits = outputs.logits_per_image

        y_pred = np.array(logits.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class PLIP_Vision_FT_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")

        self.feature_extractor = model.vision_model
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        self.feature_extractor.train()
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x)
        x = self.fc(x.pooler_output)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x)
        x = self.fc(x.pooler_output)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x)
        x = self.fc(x.pooler_output)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_Vision_FT_SSL_Entropy_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.feature_extractor = model.visual
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args
        self.labeled_names = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        probs = x.softmax(1)
        probs_u = probs[self.args.labeled_bs:]
        # labeled loss
        loss_l = nn.functional.cross_entropy(x[:self.args.labeled_bs], y[:self.args.labeled_bs])
        loss_u = -torch.mean(((probs_u.log()+1e-6)*probs_u))
        loss = loss_l + 0.0*loss_u
        acc = accuracy(x[:self.args.labeled_bs], y[:self.args.labeled_bs], topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)

        self.labeled_names += batch['img_name'][:self.args.labeled_bs]
        self.labeled_names = list(set(self.labeled_names))
        # print('labeled num: ', len(self.labeled_names))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class BMC_Vision_FT_SSL_CPS_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.feature_extractor = copy.deepcopy(model.visual)
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.feature_extractor2 = copy.deepcopy(model.visual)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args
        self.labeled_names = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        current_epoch = self.trainer.current_epoch
        x, y = batch['img'], batch['cls_label']
        x1 = self.feature_extractor(x, hidden=True)
        x1 = self.fc(x1)
        x2 = self.feature_extractor2(x, hidden=True)
        x2 = self.fc2(x2)
        probs1 = x1.softmax(1)
        probs1_u = probs1[self.args.labeled_bs:]
        pseudo1 = probs1_u.argmax(1)
        probs2 = x2.softmax(1)
        probs2_u = probs2[self.args.labeled_bs:]
        pseudo2 = probs2_u.argmax(1)
        # labeled loss
        loss_l = (nn.functional.cross_entropy(x1[:self.args.labeled_bs], y[:self.args.labeled_bs])+ \
                  nn.functional.cross_entropy(x2[:self.args.labeled_bs], y[:self.args.labeled_bs]))/2
        # pseudo labels
        loss_u = (nn.functional.cross_entropy(x2[self.args.labeled_bs:], pseudo1) + \
                nn.functional.cross_entropy(x1[self.args.labeled_bs:], pseudo2))/2
        # consistency loss
        loss_u = (torch.mean(probs1_u-probs2_u.detach()**2)+\
                    torch.mean(probs2_u-probs1_u.detach()**2))/2
        consistency_weight = self.get_current_consistency_weight(current_epoch, 30)
        # print('current_epoch:', current_epoch, consistency_weight)
        loss = loss_l + consistency_weight*loss_u
        acc = accuracy(x1[:self.args.labeled_bs], y[:self.args.labeled_bs], topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)

        self.labeled_names += batch['img_name'][:self.args.labeled_bs]
        self.labeled_names = list(set(self.labeled_names))
        # print('labeled num: ', len(self.labeled_names))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x1 = self.feature_extractor(x, hidden=True)
        x1 = self.fc(x1)
        x2 = self.feature_extractor2(x, hidden=True)
        x2 = self.fc2(x2)
        logits = (x1+x2)/2
        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_current_consistency_weight(self, current, rampup_length):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242

        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        weight = float(np.exp(-5.0 * phase * phase))
        return self.args.consistency * weight

class BMC_Vision_FT_SSL_MC_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.feature_extractor = copy.deepcopy(model.visual)
        # print(list(model.visual.children()))
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args
        self.labeled_names = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        current_epoch = self.trainer.current_epoch
        x, y = batch['img'], batch['cls_label']
        fea = self.feature_extractor(x, hidden=True)
        x1 = self.fc(fea)
        x2 = self.fc2(fea)
        probs1 = x1.softmax(1)
        probs1_u = probs1[self.args.labeled_bs:]
        pseudo1 = probs1_u.argmax(1)
        probs2 = x2.softmax(1)
        probs2_u = probs2[self.args.labeled_bs:]
        pseudo2 = probs2_u.argmax(1)
        # labeled loss
        loss_l = (nn.functional.cross_entropy(x1[:self.args.labeled_bs], y[:self.args.labeled_bs])+ \
                  nn.functional.cross_entropy(x2[:self.args.labeled_bs], y[:self.args.labeled_bs]))/2
        # pseudo labels
        loss_u = (nn.functional.cross_entropy(x2[self.args.labeled_bs:], pseudo1) + \
                nn.functional.cross_entropy(x1[self.args.labeled_bs:], pseudo2))/2
        # consistency loss
        loss_u = torch.mean((probs1_u-probs2_u)**2)
        consistency_weight = self.get_current_consistency_weight(current_epoch, 30)
        # print('current_epoch:', current_epoch, consistency_weight)
        loss = loss_l + consistency_weight*loss_u
        acc = accuracy(x1[:self.args.labeled_bs], y[:self.args.labeled_bs], topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)

        self.labeled_names += batch['img_name'][:self.args.labeled_bs]
        self.labeled_names = list(set(self.labeled_names))
        # print('labeled num: ', len(self.labeled_names))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        fea = self.feature_extractor(x, hidden=True)
        x1 = self.fc(fea)
        x2 = self.fc2(fea)
        logits = (x1+x2)/2
        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.AdamW()
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_current_consistency_weight(self, current, rampup_length):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242

        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        weight = float(np.exp(-5.0 * phase * phase))
        return self.args.consistency * weight

class BMC_LIP_Vision_FT_SSL_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        all_prompt = ["An H&E image of Adipose",'An H&E image of background',"An H&E image of debris",\
                   "An H&E image of lymphocytes", "An H&E image of mucus","An H&E image of smooth muscle",
                   "An H&E image of normal mucosa","An H&E image of cancer-associated stroma",
                   "An H&E image of adenocarcinoma epithelium"
                    ]
        text_prompt = np.array(all_prompt)[np.array(args.id_cls)]
        print(text_prompt)
        self.model = BMC_model
        self.feature_extractor = self.model.visual
        self.texts = BMC_tokenizer(text_prompt).cuda()
        # frozen text encoder, train vision encoder
        for name, param in self.model.named_parameters():
            if 'visual' not in name:
                param.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, args.num_class))
        self.args = args
        self.labeled_names = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        current_epoch = self.trainer.current_epoch
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, proj_features = self.model.visual(x, hidden=True, return_proj=True)

        # _, text_features, logit_scale = self.model(x, self.texts)
        logit_scale = self.model.logit_scale
        text_features = self.model.text(self.texts) # text features using text encoder
        logits = (logit_scale * proj_features @ text_features.t()) # LIP outputs
        # logits = proj_features @ text_features.t() # LIP outputs
        logits_fc = self.fc(image_features)
        # labeled loss
        loss_l = (nn.functional.cross_entropy(logits[:self.args.labeled_bs], y[:self.args.labeled_bs])+ \
                  nn.functional.cross_entropy(logits_fc[:self.args.labeled_bs], y[:self.args.labeled_bs]))/2
        # loss_l = nn.functional.cross_entropy(logits[:self.args.labeled_bs], y[:self.args.labeled_bs])
        # consistency loss
        probs1_u = logits.softmax(1)[self.args.labeled_bs:]
        probs2_u = logits_fc.softmax(1)[self.args.labeled_bs:]
        loss_u = torch.mean((probs1_u-probs2_u)**2)
        consistency_weight = self.get_current_consistency_weight(current_epoch, 30)
        # print('current_epoch:', current_epoch, consistency_weight)
        loss = loss_l + consistency_weight*loss_u
        acc = accuracy(logits[:self.args.labeled_bs], y[:self.args.labeled_bs], topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)

        self.labeled_names += batch['img_name'][:self.args.labeled_bs]
        self.labeled_names = list(set(self.labeled_names))
        # print('labeled num: ', len(self.labeled_names))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        image_features, proj_features = self.model.visual(x, hidden=True, return_proj=True)

        # _, text_features, logit_scale = self.model(x, self.texts)
        logit_scale = self.model.logit_scale
        with torch.no_grad():
            text_features = self.model.text(self.texts) # text features using text encoder
        logits = (logit_scale * proj_features @ text_features.t()) # LIP outputs
        logits_fc = self.fc(image_features)

        loss = nn.functional.cross_entropy(logits, y)
        
        logits = logits.softmax(1)+logits_fc.softmax(1)
        # logits = logits
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x, hidden=True)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_current_consistency_weight(self, current, rampup_length):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242

        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        weight = float(np.exp(-5.0 * phase * phase))
        return self.args.consistency * weight

class UNI_LoRA_Vision_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        ckpt = torch.load('/home/ubuntu/data/lanfz/checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin',map_location="cpu")
        # print(ckpt)
        model.load_state_dict(ckpt, strict=True)
        # for name, module in model.named_modules():
        #     print(name)
        if not pretrain:
            for name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['attn.qkv'],
            lora_dropout=0.1,
            bias="none",
        )
        lora_model = get_peft_model(model, config)
        self.feature_extractor = lora_model
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x)
        x = self.fc(x)
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        x = self.feature_extractor(x)
        x = self.fc(x)

        y_pred = np.array(x.argmax(1).cpu().detach())
        y_true = np.array(y.cpu())
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        self.log('test_f1',f1, on_step=False, on_epoch=True)
        self.log('test_recall',r, on_step=False, on_epoch=True)
        return None

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9)
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=8e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs*3//4], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



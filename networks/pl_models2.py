import lightning.pytorch as L
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
from CONCH.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from CONCH.conch.downstream.zeroshot_path import zero_shot_classifier, run_mizero


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
    print(f"Test Accuracy: {test_accuracy.item()}, f1:{f1}, precision:{p}, recall:{r}")

def cls_count(args, y_pred):
    count = [0]*args.num_class
    for item in y_pred:
        count[item] += 1
    print(count)
    return count


class CONCH_Vision_FT_Lit(L.LightningModule):
    def __init__(self, pretrain, num_class, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "/home/ubuntu/data/lanfz/checkpoints/CONCH/pytorch_model.bin")
        self.feature_extractor = model.visual
        if not pretrain:
            for params in self.feature_extractor.parameters():
                params.requires_grad = False
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_class))
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        # x = self.fc(x)
        x = self.feature_extractor(x)
        x = self.fc(x[0])

        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        # x = self.fc(x)
        x = self.feature_extractor(x)
        x = self.fc(x[0])
        loss = nn.functional.cross_entropy(x, y)
        acc = accuracy(x, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        # x = self.fc(x)
        x = self.feature_extractor(x)
        x = self.fc(x[0])
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


class CONCH_LIP_Lit(L.LightningModule):
    def __init__(self, pretrain, classnames_text, templates, id_cls, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "/home/ubuntu/data/lanfz/checkpoints/CONCH/pytorch_model.bin")
        for name, params in model.named_parameters():
            if 'visual' not in name:
                params.requires_grad = False
        self.model = model.cuda()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=device)
        self.zeroshot_weights = zeroshot_weights[:, torch.tensor(id_cls, dtype=torch.int32)]
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        # x = self.fc(x)
        image_features = self.model.encode_image(x)
        logits = image_features @ self.zeroshot_weights
        # print('------------', image_features.shape, self.zeroshot_weights.shape, logits.shape)

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        image_features = self.model.encode_image(x)
        logits = image_features @ self.zeroshot_weights
        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        # x = self.fc(x)
        image_features = self.model.encode_image(x)
        logits = image_features @ self.zeroshot_weights
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


class CONCH_LIP_LoRA_Lit(L.LightningModule):
    def __init__(self, pretrain, classnames_text, templates, id_cls, args):
        super().__init__()
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "/home/ubuntu/data/lanfz/checkpoints/CONCH/pytorch_model.bin")
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['attn.qkv'],
            lora_dropout=0.1,
            bias="none",
        )
        lora_model = get_peft_model(model, config)
        self.model = lora_model.cuda()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=device)
        self.zeroshot_weights = zeroshot_weights[:, torch.tensor(id_cls, dtype=torch.int32)]
        self.args = args

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        # x = self.fc(x)
        image_features = self.model.encode_image(x)
        logits = image_features @ self.zeroshot_weights
        # print('------------', image_features.shape, self.zeroshot_weights.shape, logits.shape)

        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log('train_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        image_features = self.model.encode_image(x)
        logits = image_features @ self.zeroshot_weights
        loss = nn.functional.cross_entropy(logits, y)
        acc = accuracy(logits, y, topk=(1,))
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc[0], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['cls_label']
        # print(x.shape)
        # x = self.feature_extractor.encode_image(x, proj_contrast=False, normalize=False)
        # x = self.fc(x)
        image_features = self.model.encode_image(x)
        logits = image_features @ self.zeroshot_weights
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
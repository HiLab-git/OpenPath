import sys
from dataset.alb_dataset2 import Tumor_dataset, Tumor_dataset_val, Tumor_dataset_val_cls, get_loader
import argparse
import torch
import os
import numpy as np
import pandas as pd
import random
from open_clip import create_model_from_pretrained, get_tokenizer
from peft import LoraConfig, TaskType, get_peft_model, get_peft_config
from transformers import CLIPProcessor, CLIPModel
from sklearn import metrics
from sklearn.cluster import KMeans
import copy


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_arguments():
    parser = argparse.ArgumentParser(
        description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=9, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--crop_size", default=224)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=512, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_type", type=str, default='combine')
    parser.add_argument("--id_cls", nargs="+", type=int, default=[3,6,8])
    parser.add_argument("--ood_cls", nargs="+", type=int, default=[0,1,2,4,5,7])
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--init_num", type=int, default=45)
    return parser.parse_args()

def get_files(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0]
    data_label = data.iloc[:, 1]
    data_label = np.array(data_label).astype(np.uint8)
    data_name = data_name.to_list()
    new_file = [{"img": img, "label": label} for img, label in zip(data_name, data_label)]
    new_dict = {k:v for k, v in zip(data_name, data_label)}
    return new_file, new_dict

def cal_acc(y_pred, y_true):
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = metrics.recall_score(y_true, y_pred, average='macro')
    print(metrics.recall_score(y_true, y_pred, average=None))
    print(f"Test Accuracy: {test_accuracy.item()}, f1:{f1}, precision:{p}, recall:{r}")

def kmean_cluster(embeds, n):
    # then Kmeans++
    cluster_learner = KMeans(n_clusters=n, init='k-means++', n_init='auto')
    cluster_learner.fit(embeds)
    cluster_idxs = cluster_learner.predict(embeds)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeds - centers)**2
    # print(embeddings.shape, centers.shape)
    # print(cluster_idxs.shape)
    dis = dis.sum(axis=1)
    # print(dis.shape)
    q_idx = np.array([np.arange(embeds.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
    return q_idx

def zero_shot_inference_random(args, train_eval_loader, id_cls, model_type='BMC'):
    # biomedCLIP
    BMC_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    BMC_model.cuda().eval()
    # print(list(BMC_model.text.named_children()))

    # PLIP
    PLIP_model = CLIPModel.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
    PLIP_processor = CLIPProcessor.from_pretrained("/home/ubuntu/data/lanfz/codes/CLIP-main/plip")
    PLIP_model.cuda().eval()

    text_prompt = ["An H&E image of Adipose",'An H&E image of background',"An H&E image of debris",\
                   "An H&E image of lymphocytes", "An H&E image of mucus","An H&E image of smooth muscle",
                   "An H&E image of normal mucosa","An H&E image of cancer-associated stroma",
                   "An H&E image of adenocarcinoma epithelium"
                    ]
    text_prompt_random = ['An H&E image of stroma', 'An H&E image of mucus',\
                          "An H&E image of fibrous", \
                          "An H&E image of squamous epithelium", "An H&E image of glandular tissue", \
                            'An H&E image of transitional epithelium', 'An H&E image of necrotic tissue', \
                            'An H&E image of Dysplasia', 'An H&E image of Hyperplasia',
                            'An H&E image of Nerves', 'An H&E image of Vessels',
                            'An H&E image of Submucosa', 'An H&E image of Inflammatory infiltrates'] # for [3, 6, 8], [6, 8]
    # text_prompt_random = ['An H&E image of mucus', "An H&E image of fibrous", \
    #                       "An H&E image of squamous epithelium", "An H&E image of glandular tissue", \
    #                         'An H&E image of transitional epithelium', 'An H&E image of necrotic tissue', \
    #                         'An H&E image of smooth muscle', 'An H&E image of Adipose'] # for [3, 6, 7, 8]
    # text_prompt_random = ["An H&E image of fibrous", "An H&E image of inflammatory cells", \
    #                       "An H&E image of squamous epithelium", "An H&E image of glandular tissue", \
    #                         'An H&E image of transitional epithelium', 'An H&E image of necrotic tissue', \
    #                         "An H&E image of cancer-associated stroma","An H&E image of adenocarcinoma epithelium"]
    text_prompt = list(np.array(text_prompt)[np.array(id_cls)])
    text_prompt += list(np.array(text_prompt_random)[:3])
    # text_prompt += [text_prompt_random[2]]
    print(text_prompt)
    with torch.no_grad():
        pred = np.zeros((args.num_class,))
        pred_all, prob_all = torch.zeros((1, )), torch.zeros((1, len(text_prompt)))
        if model_type == 'plip' or model_type == 'BMC' or model_type == 'CONCH':
            embeddings = torch.zeros((1, 512))
        if 'combine' in model_type:
            embeddings = torch.zeros((1, 1024))
        names = []
        for counter, sample in enumerate(train_eval_loader):
            x_batch = sample['img'].cuda()
            # print(x_batch.shape)
            batch_names = sample['img_name']

            if counter == 0:
                print(batch_names[0])

            # biomedclip
            if model_type == 'BMC':
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                probs = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                embeddings = torch.cat([embeddings, image_features.cpu()], dim=0)
            # plip
            if model_type == 'plip':
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch

                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                outputs = PLIP_model.forward(**inputs)
                # this is the image-text similarity score
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                image_features = outputs.image_embeds.cpu()
                embeddings = torch.cat([embeddings, image_features.cpu()], dim=0)
            if model_type == 'combine':
                # for CLIPMODEL (PLIP)
                inputs = PLIP_processor(text=text_prompt, return_tensors="pt", padding=True)
                inputs['pixel_values'] = x_batch

                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                outputs = PLIP_model.forward(**inputs)
                # this is the image-text similarity score
                logits_per_image = outputs.logits_per_image
                cur_probs1 = logits_per_image.softmax(dim=1)  
                # for open_clip model (biomedclip)
                texts = BMC_tokenizer(text_prompt).cuda()
                image_features, text_features, logit_scale = BMC_model(x_batch, texts)
                cur_probs2 = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                probs = (cur_probs1+cur_probs2)/2
                embeddings = torch.cat((embeddings, torch.cat((outputs.image_embeds.cpu(), image_features.cpu()), dim=1)), dim=0)

            logits_hard = torch.argmax(probs, dim=1)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
            prob_all = torch.cat((prob_all, probs.cpu()), dim=0)
            names += batch_names
            
    pred_all, prob_all, embeddings = pred_all[1:], prob_all[1:], embeddings[1:]
    y_pred = pred_all.numpy().astype(np.uint8)
    y_prob = prob_all

    return y_pred, y_prob, np.array(names), embeddings.clone().detach().cpu().numpy()

if __name__ == "__main__":
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    # dataset
    train_files, train_dict = get_files('/home/ubuntu/data/lanfz/datasets/CRC-VAL-HE-7K-PNG/train.csv')
    np.random.shuffle(train_files)
    
    id_cls = args.id_cls
    ood_cls = args.ood_cls
    args.num_class = len(id_cls)

    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(len(train_id_files), len(train_ood_files))
    train_files = copy.deepcopy(train_id_files) + copy.deepcopy(train_ood_files)
    np.random.shuffle(train_files)

    # train_dataset = Tumor_dataset_val(args, files=train_files[int(0.4*len(train_files)):])
    train_dataset_eval = Tumor_dataset_val_cls(args, files=train_files)
    train_eval_loader = get_loader(args, train_dataset_eval, shuffle=False)

    y_pred_raw, y_prob_raw, names, embeds = zero_shot_inference_random(args, train_eval_loader, id_cls, model_type=args.model_type)
    print(y_prob_raw.shape)

    re_id_cls = np.arange(len(id_cls))
    names_id_vlm = names[y_pred_raw<=len(id_cls)-1]
    names_id_gt = [item['img'] for item in train_id_files]

    names_idx = np.array([i for i, val in enumerate(names) if val in names_id_vlm])
    embeds_id = embeds[names_idx]

    # print(len(names_id_vlm), len(embeds_id))
    print(len(names_id_vlm), len(names_id_gt), len(list(set(names_id_gt)&set(names_id_vlm))), \
          len(list(set(names_id_gt)&set(names_id_vlm)))/len(names_id_vlm), 'vs.', len(train_id_files)/len(train_files))

    # kmeans++
    # print(embeds_id.shape)
    cluster_idx = kmean_cluster(embeds=embeds_id, n=args.init_num)
    names_init_select = names_id_vlm[cluster_idx]
    label_select = np.array([train_dict[item] for item in names_init_select])

    count = 0
    for name in names_init_select:
        if name in [item['img'] for item in train_id_files]:
            count += 1
    print('query precision: ', count/args.init_num)

    # # write pandas
    data_df = pd.DataFrame()
    data_df['img'] = names_init_select
    data_df['cls_label'] = label_select
    if args.save_csv:
        data_df.to_csv(args.save_csv, index=False)


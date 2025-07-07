import sys
from dataset.alb_dataset2 import Tumor_dataset, Tumor_dataset_val_cls, get_loader
import argparse
import torch
import os
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.cluster import KMeans
from networks.pl_models import BMC_Vision_FT_Lit
from scipy.spatial.distance import mahalanobis, euclidean, cosine
import copy
import torch.nn.functional as F
import logging
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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
    parser.add_argument("--batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--self_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_num", type=int, default=50)
    parser.add_argument("--query_times", type=int, default=5)
    parser.add_argument("--log", type=str, default='log/train_sup.txt')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--init_csv", type=str)
    parser.add_argument("--id_cls", nargs="+", type=int, default=[3,6,8])
    parser.add_argument("--ood_cls", nargs="+", type=int, default=[0,1,2,4,5,7])
    parser.add_argument("--id_ratio", type=float, default=1e-3)
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

def train_labeled(args, model, labeled_loader, save_dir):
    # monitor
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  # 监控验证集上的loss
        mode='max',  # 因为我们希望loss尽可能小，所以使用'min'
        save_top_k=1,  # 只保存最好的一个模型
        filename='best-model-{epoch:02d}-{val_acc:.3f}',  # 自定义文件名
        save_last=False,  # 是否保存最后一个epoch的模型，默认为False
    )
    trainer = L.Trainer(max_epochs=args.epochs, precision='16-mixed', check_val_every_n_epoch=args.epochs, callbacks=[checkpoint_callback],
                        logger=TensorBoardLogger(save_dir=save_dir))
    trainer.fit(model=model, train_dataloaders=labeled_loader, val_dataloaders=val_loader)

def self_training(args, model, psuedo_loader, save_dir):
    # monitor
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  # 监控验证集上的loss
        mode='max',  # 因为我们希望loss尽可能小，所以使用'min'
        save_top_k=1,  # 只保存最好的一个模型
        filename='best-model-{epoch:02d}-{val_acc:.3f}',  # 自定义文件名
        save_last=False,  # 是否保存最后一个epoch的模型，默认为False
    )
    trainer = L.Trainer(max_epochs=args.self_epochs, precision='16-mixed', check_val_every_n_epoch=args.self_epochs, callbacks=[checkpoint_callback],
                        logger=TensorBoardLogger(save_dir=save_dir))
    trainer.fit(model=model, train_dataloaders=psuedo_loader, val_dataloaders=val_loader)

@torch.no_grad()
def generate_pseudo_labels(args, model, labeled_loader, selection_loader):
    for counter, sample in enumerate(labeled_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        # cur_feature = model.feature_extractor(x_batch, hidden=True)
        cur_feature = model.feature_extractor(x_batch, hidden=True)

        if counter == 0:
            labeled_names = batch_names
            labeled_features = cur_feature
            labeled_labels = y_batch
        else:
            labeled_names += batch_names
            labeled_features = torch.cat((labeled_features, cur_feature), dim=0)
            labeled_labels = torch.cat((labeled_labels, y_batch), dim=0)

    ## labeled, split into class-wise
    for counter, cls_id in enumerate(args.id_cls):
        if counter == 0:
            class_id_embed = labeled_features[labeled_labels==cls_id].mean(0).unsqueeze(0)
        else:
            class_id_embed = torch.cat((class_id_embed,labeled_features[labeled_labels==cls_id].mean(0).unsqueeze(0)))
    # print('-----', labeled_labels)

    # unlabled
    for counter, sample in enumerate(selection_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        # cur_feature = model.feature_extractor(x_batch, hidden=True)
        cur_feature = model.feature_extractor(x_batch, hidden=True)
        cur_prob = model.fc(cur_feature).softmax(1)
        if counter == 0:
            unlabeled_probs = cur_prob
            unlabeled_labels = y_batch
            unlabeled_names = batch_names
            unlabeled_features = cur_feature
        else:
            unlabeled_probs = torch.cat((unlabeled_probs, cur_prob), dim=0)
            unlabeled_labels = torch.cat((unlabeled_labels, y_batch), dim=0)
            unlabeled_names += batch_names
            unlabeled_features = torch.cat((unlabeled_features, cur_feature), dim=0)
    distances = 1 - F.cosine_similarity(unlabeled_features.unsqueeze(1), class_id_embed.unsqueeze(0), dim=2).cpu()
    distances, unlabeled_names, unlabeled_features = np.array(distances), np.array(unlabeled_names), unlabeled_features.cpu().numpy()

    ## t-sne visualization 
    # show_tsne(labeled_features.cpu().numpy(), unlabeled_features, labeled_labels.numpy(), unlabeled_labels.numpy())

    ## Stage1: ID samples selection
    ## here, use proportion to select
    distances_min = distances.min(axis=1)
    indices = np.argsort(distances_min)
    # indices = indices[int(0.05*len(indices)):int(0.2*len(indices))]
    indices = indices[:int(args.id_ratio*len(indices))]

    candidates_features = unlabeled_features[indices]
    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]

    candidates_labels = [train_dict[item] for item in candidates_names]
    count = 0
    for item in candidates_labels:
        if int(item) in id_cls:
            count += 1
    print('ID/OOD candidates precision:', count/len(candidates_labels))

    ## candidates image, accuracy
    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    candidates_labels_re = candidates_labels.copy()
    for i in range(len(candidates_labels_re)):
        if candidates_labels_re[i] in id_cls:
            candidates_labels_re[i] = id_cls.index(candidates_labels_re[i])
        else:
            candidates_labels_re[i] = len(id_cls)
    cal_acc(candidates_preds, np.array(candidates_labels_re))

    pseudo_files = [{'img': name, 'label': label} for (name, label) in zip(candidates_names, candidates_preds)]
    print('pseudo labels size: ', len(pseudo_files))
    # return get_loader(args, Tumor_dataset(args, files=pseudo_files[:int(0.4*len(pseudo_files))]), shuffle=False, batch_size=256)
    return get_loader(args, Tumor_dataset(args, files=pseudo_files), shuffle=False, batch_size=256)

@torch.no_grad()
def sample_selection(args, model, labeled_loader, selection_loader, query_num, save_dir):
    for counter, sample in enumerate(labeled_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        # cur_feature = model.feature_extractor(x_batch, hidden=True)
        cur_feature = model.feature_extractor(x_batch, hidden=True)

        if counter == 0:
            labeled_names = batch_names
            labeled_features = cur_feature
            labeled_labels = y_batch
        else:
            labeled_names += batch_names
            labeled_features = torch.cat((labeled_features, cur_feature), dim=0)
            labeled_labels = torch.cat((labeled_labels, y_batch), dim=0)

    ## labeled, split into class-wise
    for counter, cls_id in enumerate(args.id_cls):
        if counter == 0:
            class_id_embed = labeled_features[labeled_labels==cls_id].mean(0).unsqueeze(0)
        else:
            class_id_embed = torch.cat((class_id_embed,labeled_features[labeled_labels==cls_id].mean(0).unsqueeze(0)))

    # unlabled
    for counter, sample in enumerate(selection_loader):
        x_batch = sample['img'].cuda()
        y_batch = sample['cls_label']
        batch_names = sample['img_name']
        # cur_feature = model.feature_extractor(x_batch, hidden=True)
        cur_feature = model.feature_extractor(x_batch, hidden=True)
        cur_prob = model.fc(cur_feature).softmax(1)

        if counter == 0:
            unlabeled_probs = cur_prob
            unlabeled_labels = y_batch
            unlabeled_names = batch_names
            unlabeled_features = cur_feature
        else:
            unlabeled_probs = torch.cat((unlabeled_probs, cur_prob), dim=0)
            unlabeled_labels = torch.cat((unlabeled_labels, y_batch), dim=0)
            unlabeled_names += batch_names
            unlabeled_features = torch.cat((unlabeled_features, cur_feature), dim=0)
    distances = 1 - F.cosine_similarity(unlabeled_features.unsqueeze(1), class_id_embed.unsqueeze(0), dim=2).cpu()
    distances, unlabeled_names, unlabeled_features = distances, np.array(unlabeled_names), unlabeled_features.cpu().numpy()

    ## Stage1: ID samples selection
    ## here, use proportion to select
    # print(torch.tensor(distances).shape)
    # distances entropy
    distance_probs = torch.tensor(distances).softmax(1)
    distance_entropy = -torch.sum((distance_probs.log()+1e-6)*distance_probs, dim=1)
    # distances entropy, should compute in a couple way
    distance_probs0  = torch.tensor(distances)[:,[0,1]].softmax(1)
    distance_entropy0 = -torch.sum((distance_probs0.log()+1e-6)*distance_probs0, dim=1)
    distance_probs1  = torch.tensor(distances)[:,[0,2]].softmax(1)
    distance_entropy1 = -torch.sum((distance_probs1.log()+1e-6)*distance_probs1, dim=1)
    distance_probs2  = torch.tensor(distances)[:,[1,2]].softmax(1)
    distance_entropy2 = -torch.sum((distance_probs2.log()+1e-6)*distance_probs2, dim=1)
    distance_entropy = torch.cat((distance_entropy0.unsqueeze(1),distance_entropy1.unsqueeze(1),distance_entropy2.unsqueeze(1)), dim=1)
    distance_entropy, _ = distance_entropy.max(dim=1)
    # print(distance_entropy.shape)

    distances_min = np.array(distances).min(axis=1)
    indices = np.argsort(distances_min)
    # indices = indices[int(0.05*len(indices)):int(0.2*len(indices))]
    indices = indices[:int(args.id_ratio*len(indices))]

    candidates_features = unlabeled_features[indices]
    candidates_names = unlabeled_names[indices]
    candidates_probs = unlabeled_probs[indices]
    candidates_distance_entropy = distance_entropy[indices]

    candidates_labels = [train_dict[item] for item in candidates_names]
    count = 0
    for item in candidates_labels:
        if int(item) in id_cls:
            count += 1
    print('ID/OOD candidates precision:', count/len(candidates_labels))

    ## candidates image, accuracy
    candidates_preds = candidates_probs.argmax(1).cpu().numpy()
    candidates_labels_re = candidates_labels.copy()
    for i in range(len(candidates_labels_re)):
        if candidates_labels_re[i] in id_cls:
            candidates_labels_re[i] = id_cls.index(candidates_labels_re[i])
        else:
            candidates_labels_re[i] = len(id_cls)
    cal_acc(candidates_preds, np.array(candidates_labels_re))

    ## Kmeans selection
    cluster_idx = kmean_cluster(embeds=candidates_features, n=query_num)
    # print(len(cluster_idx))
    selected_names = np.array(candidates_names)[cluster_idx]
    selected_labels = [train_dict[item] for item in selected_names]
    
    ## save selection sample
    data_df = pd.DataFrame()
    data_df['img'] = selected_names
    data_df['cls_label'] = np.array(selected_labels)
    data_df.to_csv(save_dir, index=False)

    count = 0
    for name in selected_names:
        if name in [item['img'] for item in train_id_files]:
            count += 1
    print('query precision: ', count/args.query_num)
    return count/args.query_num

if __name__ == "__main__":
    args = get_arguments()
    seed_torch(args.seed)
    torch.cuda.set_device(args.gpu[0])

    l = logging.getLogger(__name__)
    fileHandler = logging.FileHandler(args.log, mode='a')
    l.setLevel(logging.INFO)
    l.addHandler(fileHandler)

    # dataset
    train_files, train_dict = get_files('/home/ubuntu/data/lanfz/datasets/CRC-VAL-HE-7K-PNG/train.csv')
    np.random.shuffle(train_files)
    train_all_files = train_files[:]
    test_files, _ = get_files('/home/ubuntu/data/lanfz/datasets/CRC-VAL-HE-7K-PNG/test.csv')
    np.random.shuffle(test_files)

    cls_count(args, y_pred=np.array([itme['label'] for itme in train_files]))

    id_cls = args.id_cls
    ood_cls = args.ood_cls
    args.num_class = len(id_cls)

    # here, OOD detection
    train_id_files = [item for item in train_files if item['label'] in id_cls]
    train_ood_files = [item for item in train_files if item['label'] in ood_cls]
    print(len(train_id_files), len(train_ood_files))

    test_id_files = [item for item in test_files if item['label'] in id_cls]
    test_ood_files = [item for item in test_files if item['label'] in ood_cls]
    print(len(test_id_files), len(test_ood_files))

    # update train_files
    train_files = copy.deepcopy(train_id_files)+copy.deepcopy(train_ood_files)

    # train_id_files = train_id_files[:10000]
    val_id_files = copy.deepcopy(test_id_files)[:]
    for item in val_id_files: 
        item['label'] = id_cls.index(item['label'])

    ## labeled loader and selection loader
    initial_data = pd.read_csv(args.init_csv)
    initial_names = initial_data.iloc[:, 0].to_numpy()
    initial_labeled = [item for item in train_files if item['img'] in initial_names]
    
    labeled_ID = [item for item in train_id_files if item['img'] in initial_names]
    candidates = [item for item in train_files if item['img'] not in initial_names]
    # np.random.shuffle(candidates)
    candidates = copy.deepcopy(candidates)[:]
    print('initial labeled size: ', len(labeled_ID), 'candidates size: ', len(candidates))
    labeled_dataset = Tumor_dataset_val_cls(args, files=initial_labeled)
    selection_dataset = Tumor_dataset_val_cls(args, files=candidates)
    # selection_dataset = Tumor_dataset(args, files=candidates) # add purturbations
    # selection_dataset = Tumor_dataset_two_weak_rcc(args, files=candidates) # add more purturbations
    val_dataset = Tumor_dataset_val_cls(args, files=val_id_files)
    labeled_loader = get_loader(args, labeled_dataset, shuffle=True)
    selection_loader = get_loader(args, selection_dataset, shuffle=False)
    val_loader = get_loader(args, val_dataset, shuffle=False)

    # labeled_loader_train, need to use ID samples
    labeled_ID = copy.deepcopy(labeled_ID)
    for item in labeled_ID: 
        item['label'] = id_cls.index(item['label'])
    # print(labeled[:10])
    cls_count(args, y_pred=np.array([itme['label'] for itme in labeled_ID]))
    # print(len(labeled_ID), [item['label'] for item in labeled_ID])
    labeled_loader_train = get_loader(args, Tumor_dataset(args, files=labeled_ID), shuffle=True)

    query_times = args.query_times
    precision_list = []
    for i in range(query_times):
    # for i in [1, 2, 3, 4]:
        # initialization
        if i == 0:
            model = BMC_Vision_FT_Lit(pretrain=True, num_class=args.num_class, args=args)
            save_dir = 'lightning_logs/CRC100K-BMC-ST-0-L/'
            train_labeled(args, model, labeled_loader_train, save_dir=save_dir)
            # load trained model
            ckpt_path = save_dir + 'lightning_logs/version_0/checkpoints/' + os.listdir(save_dir+'lightning_logs/version_0/checkpoints')[0]
            # print(ckpt_path)
            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            pseudo_loader = generate_pseudo_labels(args, model, labeled_loader, selection_loader)
            save_dir = 'lightning_logs/CRC100K-BMC-ST-0-ST/'
            self_training(args, model, pseudo_loader, save_dir)
            labeled_data_all = copy.deepcopy(initial_labeled)
            precision_list.append(len(labeled_ID)/50)
        else:
            save_dir = f'lightning_logs/CRC100K-BMC-ST-{i-1}-ST/'
            ckpt_path = save_dir + 'lightning_logs/version_0/checkpoints/' + os.listdir(save_dir+'lightning_logs/version_0/checkpoints')[0]
            # print(ckpt_path)
            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            save_csv = f'al_file/BMC_query{i}_labeled.csv'
            cur_precision = sample_selection(args, model, labeled_loader, selection_loader, query_num=args.query_num, save_dir=save_csv)
            save_dir = f'lightning_logs/CRC100K-BMC-ST-{i}-L/'
            labeled_data, _ = get_files(save_csv)
            labeled_data_all += labeled_data
            labeled_data_ID = [item for item in labeled_data if item['label'] in id_cls] # train on currrent samples
            # labeled_data_ID = [item for item in labeled_data_all if item['label'] in id_cls] # train on currrent samples
            labeled_data_ID = copy.deepcopy(labeled_data_ID)
            for item in labeled_data_ID: 
                item['label'] = id_cls.index(item['label'])
            print(len(labeled_data_ID))
            labeled_loader_ID = get_loader(args, Tumor_dataset(args, labeled_data_ID))
            train_labeled(args, model, labeled_loader_ID, save_dir)
            ## update labeled_loader, selection_loader
            ckpt_path = save_dir + 'lightning_logs/version_0/checkpoints/' + os.listdir(save_dir+'lightning_logs/version_0/checkpoints')[0]
            # print(ckpt_path)
            labeled_names_all = [item['img'] for item in labeled_data_all]
            candidates = [item for item in train_files if item['img'] not in copy.deepcopy(labeled_names_all)]
            print('candidates size: ', len(candidates))
            print(len(labeled_data_all), len(candidates))
            model = BMC_Vision_FT_Lit.load_from_checkpoint(checkpoint_path=ckpt_path, pretrain=True, num_class=len(id_cls), args=args)
            labeled_loader = get_loader(args, Tumor_dataset_val_cls(args, labeled_data_all))
            selection_loader = get_loader(args, Tumor_dataset_val_cls(args, candidates))
            pseudo_loader = generate_pseudo_labels(args, model, labeled_loader, selection_loader)
            save_dir = f'lightning_logs/CRC100K-BMC-ST-{i}-ST/'
            self_training(args, model, pseudo_loader, save_dir)
            # metrics
            precision_list.append(cur_precision)
        print(precision_list)
    # record
    np.savetxt('log/Ours_precision.txt', np.array(precision_list),delimiter=',')
            
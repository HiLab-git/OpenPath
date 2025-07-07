import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util.two_stream import TwoStreamBatchSampler


class Tumor_dataset(Dataset):
    def __init__(self, args, files):
        super().__init__()
        """
        args: params
        files: [{}, {}], "img", "label"
        """
        self.files = files
        # A.RandomRotate90, A.RandomGridShuffle(), A.ColorJitter(), A.GaussianBlur(), A.Sharpen(), A.RandomCrop(), A.Flip(), A.Affine(), A.ElasticTransform()
        
        self.train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.train_transform(image=image)["image"]

        return {"img":image, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_aug(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_w = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
            ]
        )
        self.train_transform_s = A.Compose(
            [
                A.GaussianBlur(p=0.5),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.CoarseDropout(p=0.5),
                # A.GridDistortion(p=1),
                # A.ElasticTransform(p=1),
                A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
            ]
        )
        self.trans_tensor = A.Compose([A.Normalize(), ToTensorV2()])
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_w = self.train_transform_w(image=image)["image"]
        image_s = self.train_transform_s(image=image_w)["image"]
        image_w = self.trans_tensor(image=image_w)["image"]
        image_s = self.trans_tensor(image=image_s)["image"]

        return {"img_w":image_w, "img_s":image_s, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_two_weak(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_w1 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                A.Normalize(),
                ToTensorV2()
            ]
        )
        self.train_transform_w2 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                A.Normalize(),
                ToTensorV2()
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_w1 = self.train_transform_w1(image=image)["image"]
        image_w2 = self.train_transform_w2(image=image)["image"]

        return {"img1":image_w1, "img2":image_w2, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_two_strong(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_s1 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                A.GaussianBlur(p=0.5),
                A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
                A.Normalize(),
                ToTensorV2()
            ]
        )
        self.train_transform_s2 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
                A.RandomRotate90(),
                A.Flip(),
                A.GaussianBlur(p=0.5),
                A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
                A.Normalize(),
                ToTensorV2()
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_w1 = self.train_transform_s1(image=image)["image"]
        image_w2 = self.train_transform_s2(image=image)["image"]

        return {"img1":image_w1, "img2":image_w2, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)

class Tumor_dataset_aug3(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_w = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomRotate90(),
                A.Flip(),
            ]
        )
        self.train_transform_s = A.Compose(
            [
                A.GaussianBlur(p=1),
                A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=1),
            ]
        )
        self.train_transform_s2 = A.GaussianBlur(p=1)
        self.train_transform_s3 = A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=1)
        self.trans_tensor = A.Compose([A.Normalize(), ToTensorV2()])
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_w = self.train_transform_w(image=image)["image"]
        tmp = np.random.rand()
        if  tmp < 0.4:
            image_s = self.train_transform_s2(image=image_w)["image"]
        elif tmp > 0.8:
            image_s = self.train_transform_s(image=image_w)["image"]
        else:
            image_s = self.train_transform_s3(image=image_w)["image"]
        image_w = self.trans_tensor(image=image_w)["image"]
        image_s = self.trans_tensor(image=image_s)["image"]

        return {"img_w":image_w, "img_s":image_s, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_aug2(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_w = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomRotate90(),
            ]
        )
        self.train_transform_s = A.Compose(
            [
                A.RandomResizedCrop(scale=(0.5, 1), height=args.crop_size, width=args.crop_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomRotate90(),
                # A.GaussianBlur(p=0.5),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.CoarseDropout(p=0.5),
                # A.GridDistortion(p=1),
                # A.ElasticTransform(p=1),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
            ]
        )
        self.trans_tensor = A.Compose([A.Normalize(), ToTensorV2()])
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_w = self.train_transform_w(image=image)["image"]
        image_s = self.train_transform_s(image=image)["image"]
        image_w = self.trans_tensor(image=image_w)["image"]
        image_s = self.trans_tensor(image=image_s)["image"]

        return {"img_w":image_w, "img_s":image_s, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)

class Tumor_dataset_dual_stream(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform_w = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomRotate90(),
            ]
        )
        
        self.train_transform_s = A.Compose(
            [
                A.CoarseDropout(p=0.5),
                A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
            ]
        )
        self.trans_tensor = A.Compose([A.Normalize(), ToTensorV2()])
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        image_w1 = self.train_transform_w(image=image)["image"]
        image_s1 = self.train_transform_s(image=image_w1)["image"]
        image_w1 = self.trans_tensor(image=image_w1)["image"]
        image_s1 = self.trans_tensor(image=image_s1)["image"]
    
        image_w2 = self.train_transform_w(image=image)["image"]
        image_s2 = self.train_transform_s(image=image_w2)["image"]
        image_w2 = self.trans_tensor(image=image_w2)["image"]
        image_s2 = self.trans_tensor(image=image_s2)["image"]

        return {"img_w1":image_w1, "img_s1":image_s1, "img_w2":image_w2, "img_s2":image_s2, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)
    

class Tumor_dataset_val(Dataset):
    def __init__(self, args, files):
        super().__init__()
        self.files = files
        self.train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                # A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.train_transform(image=image)["image"]

        return {"img":image, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)


class Tumor_dataset_taal(Dataset):
    def __init__(self, args, files):
        super().__init__()
        """
        args: params
        files: [{}, {}], "img", "label"
        """
        self.files = files
        # A.RandomRotate90, A.RandomGridShuffle(), A.ColorJitter(), A.GaussianBlur(), A.Sharpen(), A.RandomCrop(), A.Flip(), A.Affine(), A.ElasticTransform()
        
        self.train_transform1 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                # A.GaussianBlur(),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.train_transform2 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                # A.GaussianBlur(),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.train_transform3 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                # A.GaussianBlur(),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image1 = self.train_transform1(image=image)["image"]
        image2 = self.train_transform2(image=image)["image"]
        image3 = self.train_transform3(image=image)["image"]

        return {"img1":image1, "img2":image2, "img3":image3, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)
    
class Tumor_dataset_cross(Dataset):
    def __init__(self, args, files):
        super().__init__()
        """
        args: params
        files: [{}, {}], "img", "label"
        """
        self.files = files
        # A.RandomRotate90, A.RandomGridShuffle(), A.ColorJitter(), A.GaussianBlur(), A.Sharpen(), A.RandomCrop(), A.Flip(), A.Affine(), A.ElasticTransform()
        
        self.trans_w1 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.trans_w2 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.trans_s1 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                A.GaussianBlur(),
                A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.trans_s2 = A.Compose(
            [
                A.SmallestMaxSize(max_size=args.input_size),
                A.RandomCrop(height=args.crop_size, width=args.crop_size),
                A.RandomRotate90(),
                A.Flip(),
                A.GaussianBlur(),
                A.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )      
    def __getitem__(self, index):
        cur_item = self.files[index]
        image_path = cur_item['img']
        image_label = cur_item['label']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_w1 = self.trans_w1(image=image)["image"]
        image_w2 = self.trans_w2(image=image)["image"]
        image_s1 = self.trans_s1(image=image)["image"]
        image_s2 = self.trans_s2(image=image)["image"]

        return {"img_w1":image_w1, "img_w2":image_w2, "img_s1":image_s1, "img_s2":image_s2, "label":image_label, "img_name":image_path}

    def __len__(self):
        return len(self.files)
    

def get_loader(args, ds):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    return loader

def get_train_loader_ssl(args, ds, labeled_idxs, unlabeled_idxs):
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    train_loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=False
    )
    return train_loader


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
from easydict import EasyDict
from networks import Core
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from flags import args

class ImagenetSubset(TensorDataset):
    def __init__(self, dataset):
        """
        dataset: The training imagenet created with ImageFolder
    
        Classes:
        '0': n01440764 - 'tench',
        '1': n02102040 - 'English springer',
        '2': n02979186 - 'cassette player',
        '3': n03000684 - 'chain saw',
        '4': n03028079 - 'church',
        '5': n03394916 - 'French horn',
        '6': n03417042 - 'garbage truck',
        '7': n03425413 - 'gas pump',
        '8': n03445777 - 'golf ball',
        '9': n03888257 - 'parachute'
        """
        super(ImagenetSubset, self).__init__()
        self.dataset = dataset
        self.labels = np.array(self.dataset.targets)
        self.classes = np.array([1, 3, 4, 5, 6])
        self.indices = []
        self.new_targets = []
        
        for i, cls in enumerate(self.classes):
            where = np.where(self.labels==cls)[0]
            self.indices.append(where)
            self.new_targets.append(i*np.ones(len(where)))
        
        self.new_targets = np.concatenate(self.new_targets)
        self.all_idx = np.concatenate(self.indices)

        self.class_idx = []
        for i in range(len(self.classes)):
            self.class_idx.append(np.where(self.new_targets==i)[0])
            
    def __getitem__(self, idx):
        idx = self.all_idx[idx]
        img, label = self.dataset[idx]
        label = np.where(self.classes==label)[0].squeeze()
        return img, torch.tensor(label).long()
                    
    def __len__(self):
        return len(self.all_idx)

class MaskedSet(TensorDataset):
    """
    Masked Dataset where you get img, lab, and model(img)
    """
    def __init__(self, dataset, model_outs):
        super(MaskedSet, self).__init__()
        self.dataset = dataset
        self.model_outs = model_outs
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, self.model_outs[idx]
                    
    def __len__(self):
        return len(self.dataset)

class Loaders:
    
    """
    
    transform: with a Random resized crop, random horizontal flips, 
    Grayscale, and normalization as base transforms, we have the additional choices
    
    'n': base
    'sn': random scales
    'tn': random translations
    'rn': random rotations
    
    """
    
    def __init__(
        self, 
        batch_size: int = 1, 
        class_portion: float = 1., 
        shuffle_train: bool = False,
        shuffle_test: bool = False,
        num_workers: int = 0,
        transform: str = 'n',
        pretrained_model=None,
        architecture='vgg'
    ):
        super().__init__()
        
        assert class_portion <= 1. and class_portion > 0., 'class_portion should be a float between 0 & 1'
        assert transform in ['n', 'sn', 'tn', 'rn']
        assert architecture in ['vgg', 'resnet']
        
        self.batch_size = batch_size
        self.class_portion = class_portion
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test
        self.num_workers = num_workers
        self.pretrained_model = pretrained_model
        self.architecture = architecture

        RA = {
            'n': transforms.RandomAffine(degrees=0.),
            'sn': transforms.RandomAffine(degrees=0., scale=(0.5, 1.5)),
            'tn': transforms.RandomAffine(degrees=0., translate=(0.4, 0.4)),
            'rn': transforms.RandomAffine(degrees=180.)
        }

        self.train_transforms = transforms.Compose([
            transforms.Resize(args.img_size[0]),
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            RA[transform],
            transforms.ToTensor(),
            transforms.Normalize(*args.data_stats)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(args.img_size[0]),
            transforms.CenterCrop(args.img_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(*args.data_stats)
        ])

    def trainset(self):
        trainpath = args.training_data
        trainset = ImageFolder(root=trainpath, transform=self.train_transforms)
        return ImagenetSubset(trainset)

    def testset(self):
        testpath = args.test_data
        testset = ImageFolder(root=testpath, transform=self.test_transforms)
        return ImagenetSubset(testset)
    
    def get_model_outs(self, dataset):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        core = Core(num_classes=5, architecture=self.architecture)
        checkpoint = torch.load(self.pretrained_model, map_location="cpu")
        core.load_state_dict(checkpoint['state_dict'])
        core = core.to(device)
        core.eval()
        model_outs = torch.zeros(len(loader.dataset), 5)
        print('Getting Symmetry Invariance Targets')

        with torch.no_grad():
            for i, (inputs, targets) in tqdm(enumerate(loader)):
                out = core(inputs.to(device))
                model_outs[i] = out.squeeze().detach().cpu()

        return model_outs
    
    def trainsubset(self):
        trainset = self.trainset()
        
        if self.class_portion < 1.:
            train_cl_idx = trainset.class_idx
            train_idx = []
            for i in range(len(trainset.classes)):
                train_idx.append(
                    np.random.choice(
                        train_cl_idx[i], 
                        int(self.class_portion * len(train_cl_idx[i])),
                        replace=False
                    )
                )
            train_idx = np.concatenate(train_idx)
            trainset = Subset(trainset, train_idx)

        if self.pretrained_model:
            model_outs = self.get_model_outs(trainset)
            trainset = MaskedSet(trainset, model_outs)
            
        return trainset
    
    def testsubset(self):
        testset = self.testset()
        
        if self.class_portion < 1.:
            test_cl_idx = testset.class_idx
            test_idx = []
            for i in range(len(testset.classes)):
                test_idx.append(
                    np.random.choice(
                        test_cl_idx[i], 
                        int(self.class_portion * len(test_cl_idx[i])),
                        replace=False
                    )
                )
            test_idx = np.concatenate(test_idx)
            testset = Subset(testset, test_idx)
        
        if self.pretrained_model:
            model_outs = self.get_model_outs(testset)
            testset = MaskedSet(testset, model_outs)
            
        return testset
    
    def trainloader(self):
        return DataLoader(
            self.trainsubset(), 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_train, 
            num_workers=self.num_workers
        )
    
    def testloader(self):
        return DataLoader(
            self.testsubset(), 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_test, 
            num_workers=self.num_workers
        )
    
class MaskedImages(TensorDataset):
    def __init__(self, single_img_data, single_img_masks, mask=None, adv=False):
        
        self.single_img_data = single_img_data
        self.single_img_masks = single_img_masks
        
        self.mask = mask
        if self.mask: assert mask in ['standard', 'complement']
        self.adv = adv
        
    def __len__(self):
        return len(self.single_img_data.images)

    def __getitem__(self, idx):
        mask = self.single_img_masks.masks[idx].squeeze(0)
        if self.adv:
            img = self.single_img_data.adv_images[idx]
        else:
            img = self.single_img_data.images[idx]
            
        img = torch.fft.fft2(img.squeeze(0))
        if self.mask=='complement':
            mask[mask > 1e-8] = 1.
            img = (1. - mask) * img
        elif self.mask=='standard':
            img = mask * img
        
        img = torch.fft.ifft2(img).real
        label = self.single_img_data.predictions[idx].item()
        return img, label
    
class MaskDataset(TensorDataset):
    def __init__(self, single_img_masks, adv_labels=False):
        self.single_img_masks = single_img_masks
        self.adv_labels = adv_labels
        
    def __len__(self):
        return len(self.single_img_masks.masks)

    def __getitem__(self, idx):
        mask = self.single_img_masks.masks[idx]
        mask = torch.fft.fftshift(mask)
        if self.adv_labels:
            label = torch.randint(low=0, high=5, size=(1,)).item()
        else:
            label = self.single_img_masks.final_predictions[idx].item()
        return mask, label
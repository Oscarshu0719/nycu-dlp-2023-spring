from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

'''===============================================================
1. Title:     

DLP spring 2023 Lab7 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]

==============================================================='''


class ResNet(object):
    def __init__(self, 
            num_classes: int, dir_dataset: str, device: str, path_ckpt='checkpoint.pth'):
        self.device = device
        
        self.build_model()
        self.load_model(Path(dir_dataset, path_ckpt))
        
        self.model.eval()
        self.num_classes = num_classes
        
    def build_model(self) -> None:
        self.model = models.resnet18().to(self.device)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid()
        )
        
    def load_model(self, path_ckpt: str) -> None:
        ckpt = Path(path_ckpt)
        try: 
            self.model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
            self.model = self.model.to(self.device)
            
            print(f'Finished loading evaluation model from {ckpt} ...')
        except Exception as msg:
            print(f'Failed to load checkpoint from {ckpt}. Message: \n{msg}')
        
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
                    
        return acc / total
    
    def eval(self, images, labels):
        with torch.no_grad():
            out = self.model(images)
            acc = self.compute_acc(out.to('cpu'), labels.to('cpu'))
            
            return acc
        
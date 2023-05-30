def get_train_loader(
        batch_size: int, dir_dataset: str, 
        num_workers=8, shuffle=True) -> tuple[DataLoader, int]: 
    dataset = _ClevrTrainingDataset(dir_dataset)
        
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle), \
    dataset.num_classes, dataset.data_shape
    
def get_test_labels(dir_dataset: str) -> list:
    with open(Path(dir_dataset, 'objects.json'), 'r') as file:
        classes = json.load(file)
        
    with open(Path(dir_dataset, 'test.json'), 'r') as file:
        conds_list = json.load(file)
        
    labels = torch.zeros(len(conds_list), len(classes))
    for i, conds in enumerate(conds_list):
        for cond in conds:
            labels[i, int(classes[cond])] = 1.

    return labels

class _ClevrTrainingDataset(Dataset):
    def __init__(self, dir_dataset: str, dir_root='iclevr'):
        self.dir_dataset = dir_dataset
        self.dir_root = dir_root
        
        with open(Path(dir_dataset, 'objects.json'), 'r') as file:
            self.classes = json.load(file)
        self.num_classes = len(self.classes)
        
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        
        self.max_objects = 0
        self.img_names = []
        self.img_conds = []
        
        with open(Path(dir_dataset, 'train.json'), 'r') as file:
            dict = json.load(file)
            for name, conds in dict.items():
                self.img_names.append(name)
                self.max_objects = max(self.max_objects, len(conds))
                self.img_conds.append([self.classes[cond] for cond in conds])
                
        img, _ = self.__getitem__(0)
        self.data_shape = img.shape
                
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img = Image.open(
            Path(self.dir_dataset, self.dir_root, self.img_names[index])).convert('RGB')
        img = self.transforms(img)
        cond = self.int2one_hot(self.img_conds[index])
        
        return img, cond

    def int2one_hot(self,int_list):
        one_hot = torch.zeros(self.num_classes)
        for i in int_list:
            one_hot[i] = 1.
            
        return one_hot
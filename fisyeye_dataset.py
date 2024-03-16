from torch.utils.data import Dataset
import os
import cv2
import torch

class FishEyeDataset(Dataset):
    def __init__(self, root='Fisheye', transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'labels')
        self.image_files = os.listdir(self.image_dir)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = cv2.imread(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        label_path = os.path.join(self.label_dir, self.image_files[index].replace('.jpg', '.txt'))
        with open(label_path) as f:
            labels = f.readlines()
        
        targets = {
            'labels': [],
            'boxes': []
        }
        for label in labels:
            label = label.split()
            class_id = int(label[0])
            x_center, y_center, width, height = map(float, label[1:])
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            targets['labels'].append(class_id)
            targets['boxes'].append([x_min, y_min, x_max, y_max])
        
        targets['labels'] = torch.LongTensor(targets['labels'])
        targets['boxes'] = torch.FloatTensor(targets['boxes'])
        
        return image, targets

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    dataset = FishEyeDataset(root='Fisheye')

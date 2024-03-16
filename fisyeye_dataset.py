from torch.utils.data import Dataset
import os
import cv2
import torch
class FishEyeDataset(Dataset):
    def __init__(self,root = 'Fisheye',transform = None) -> None:
        self.root = root
        self.transform = transform
        self.image_files = os.listdir(os.path.join(self.root, 'images'))

    def __getitem__(self, index):
        image = cv2.imread(os.path.join('Fisheye\images',os.listdir(os.path.join(self.root,'images'))[index]))
        if self.transform:
            image = self.transform(image)
        with open(os.path.join('Fisheye\labels',os.listdir(os.path.join(self.root,'labels'))[index])) as f:
            label = f.readlines()
        labels = []
        boxes = []
        for i in label:
            labels.append(int(i[0]))
            i = i.split()
            box = []
            for a in i[1:]:
                box.append(float(a))
            x_center, y_center, width, height = box  
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            boxes.append([x_min,y_min,x_max,y_max])
        target = {
            'labels' : torch.LongTensor(labels),
            'boxes' : torch.FloatTensor(boxes)
        }
        return image,target
    def __len__(self):
        return len(self.image_files)
if __name__ == '__main__':
    print('hello')
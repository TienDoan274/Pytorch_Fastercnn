from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import io
import torch
import torchvision
import cv2
from Source.fisyeye_dataset import FishEyeDataset
from torchvision.transforms import ToTensor
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
import argparse
import os
import shutil
from tqdm.autonotebook import tqdm
def get_args():
    parser = argparse.ArgumentParser(description='train_fisheye_with_fasterRCNN')
    parser.add_argument('--data_path',type=str,default='Fisheye')
    parser.add_argument('--learning_rate','-lr',type=int,default=1e-2)
    parser.add_argument('--checkpoint_path','-ckp',type=str,default=None)
    parser.add_argument('--tensorboard_path',type=str,default='tensorboard')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--save_path',type=str,default='saved_models')
    parser.add_argument('--epochs',type=int,default=50)
    args = parser.parse_args()
    return args
def collate_fn(batch):
    images,targets = zip(*batch)
    return list(images),list(targets)
def train(args):
    os.makedirs(args.save_path,exist_ok=True)
    epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device)
    train_data = FishEyeDataset(root=args.data_path,transform=ToTensor())
    val_data = FishEyeDataset(root=args.data_path,transform=ToTensor())

    train_loader = DataLoader(train_data,batch_size=args.batch_size,num_workers=2,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_data,batch_size=args.batch_size,num_workers=2,collate_fn=collate_fn)
    optimizer = SGD(model.parameters(),lr = args.learning_rate)
    if args.tensorboard_path :
        if os.path.isdir(args.tensorboard_path):
            shutil.rmtree(args.tensorboard_path)
        os.makedirs(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        ckp = torch.load(args.checkpoint_path)
        start_epoch = ckp['last_epoch']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
    else:
        start_epoch = 0
    metric = MeanAveragePrecision(iou_type='bbox')
    best_map = 0
    for epoch in range(start_epoch,start_epoch+epochs):
        model.train()
        progress_bar = tqdm(train_loader,colour='cyan')
        for i,(images,targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            targets = [{'boxes':target['boxes'].to(device),'labels':target['labels'].to(device)} for target in targets]
            output = model(images,targets)
            total_loss = sum(loss for loss in output.values())
            writer.add_scalar("Train/loss",total_loss,epoch*len(train_loader)+i)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, start_epoch+epochs, total_loss))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        val_progress_bar = tqdm(val_loader,colour='yellow')
        for i,(images,targets) in enumerate(val_progress_bar):
            images = [image.to(device) for image in images]
            targets = [{'boxes':target['boxes'].to(device),'labels':target['labels'].to(device)} for target in targets]
            with torch.no_grad():
                predict = model(images)
            metric.update(predict,targets)
        map = metric.compute()
        writer.add_scalar("Val/mAP", map["map"], epoch)
        checkpoint = {
            'model': model.state_dict(),
            'last_epoch': epoch+1,
            'optimizer': optimizer.state_dict(),
            'map_score': map['map']
        }
        torch.save(checkpoint,os.path.join(args.save_path,'last.pt'))
        if map['map']>best_map:
            torch.save(checkpoint,os.path.join(args.save_path,'last.pt'))
            best_map=map['map']
            
        

if __name__ == '__main__':
    args =get_args()
    train(args)
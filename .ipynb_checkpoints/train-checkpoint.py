from PIL import Image
import os
import math
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

from mobilevit import MobileViT
from arcface import ArcFace


# training params
EPOCHS = 50
BATCH_SIZE = 128
BASE_LR = 2e-3
WARMUP_ITERS = 200
MIN_LR = 0.1
ARCH = 'x_small'

class PalmDataset(Dataset):
    def __init__(self, img_dir, img_list, transform=None):
        super(PalmDataset, self).__init__()
        self.img_dir = img_dir
        with open(os.path.join(self.img_dir, img_list)) as f:
            lines = f.readlines()
        self.images = [line.rstrip('\n') for line in lines]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path)
        label = (int(img_file[:3])-1) * 2 + int(img_file[8] == 'r')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


class MobileVitClassifier(nn.Module):
    """MobileVit backbone + classifier
    """
    def __init__(self, num_cls, arch='small', last_channels=1024, gd_conv=True):
        super(MobileVitClassifier, self).__init__()
        self.net = MobileViT(arch=arch,
                             last_channels=last_channels,
                             gd_conv=gd_conv)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # feat_dim = {'xx_small': 320,
        #              'x_small': 384,
        #              'small': 640}
        # self.weight = nn.Parameter(torch.zeros(num_cls, feat_dim[arch]))
        self.weight = nn.Parameter(torch.zeros(num_cls, last_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        # print(x.shape)
        feat = self.net(x)
        # feat = self.pool(feat).flatten(1)
        # print(feat.shape)
        norm_feat = (feat**2).sum(axis=1, keepdim=True).sqrt()
        feat = feat / norm_feat
        norm_weight = (self.weight**2).sum(axis=1, keepdim=True).sqrt()
        weight = self.weight / norm_weight
        logits = F.linear(feat, weight)
        return logits


def get_lr_scheduler(optimizer, max_iters, warmup_iters, min_lr=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return current_step / warmup_iters
        else:
            progress = (current_step - warmup_iters) / (max_iters - warmup_iters)
            return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)



if __name__ == '__main__':
    # dataset and transforms
    train_transform = transforms.Compose([
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.RandomResizedCrop((256, 256), scale=(0.75, 1.25), ratio=(0.75, 1.25)),
        transforms.ColorJitter(0.25, 0.1, 0.1, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    training_data = PalmDataset(img_dir='/root/autodl-tmp/verification/ROI',
                                img_list='train.txt',
                                transform=train_transform)
    test_data = PalmDataset(img_dir='/root/autodl-tmp/verification/ROI',
                            img_list='test.txt',
                            transform=test_transform)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    model = MobileVitClassifier(num_cls=360, arch=ARCH, last_channels=1024, gd_conv=True).cuda()
    arc_face = ArcFace().cuda()
    loss_func = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=BASE_LR)
    num_iters = len(train_dataloader) * EPOCHS
    lr_scheduler = get_lr_scheduler(optimizer, num_iters, WARMUP_ITERS, MIN_LR)
    
    eval_loss_func = nn.CrossEntropyLoss(reduction='sum').cuda()

    best_acc = 0
    best_eval_loss = 10000
    
    directory_name = 'output' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(os.getcwd(), directory_name)
    os.makedirs(output_path)

    for epoch in range(EPOCHS):
        train_correct = train_total = 0
        model.train()
        for image, target in train_dataloader:
            image = image.cuda()
            target = target.cuda()

            # forward
            logits = model(image)
            with torch.no_grad():
                train_correct += (logits.argmax(axis=1) == target).sum().item()
                train_total += target.shape[0]
            logits = arc_face(logits, target)
            loss = loss_func(logits, target)
            print(f"Loss: {loss.item()}")

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        print(f"[Train] Epoch {epoch + 1}, accuracy {train_correct / train_total}")
        torch.save(model.net.state_dict(), f'{output_path}/{ARCH}_model_weights_epoch_{epoch + 1}.pth')

        # eval
        eval_correct = eval_total = 0
        eval_loss = 0
        model.eval()
        for image, target in test_dataloader:
            image = image.cuda()
            target = target.cuda()
            with torch.no_grad():
                logits = model(image)
                eval_loss += eval_loss_func(logits, target).cpu().item()
                eval_correct += (logits.argmax(axis=1) == target).sum().item()
                eval_total += target.shape[0]
        eval_acc = eval_correct / eval_total
        eval_loss = eval_loss / eval_total
        
        print(f"[Eval] Epoch {epoch + 1}, loss {eval_loss:.6f}, accuracy {eval_acc:.6f}")
        
        if eval_acc > best_acc:
            torch.save(model.net.state_dict(), f'{output_path}/{ARCH}_model_weights_best.pth')
            print(f'Model saved as {ARCH}_model_weights_best.pth')
            best_acc = eval_acc
        elif eval_acc == best_acc and eval_loss < best_eval_loss:
            torch.save(model.net.state_dict(), f'{output_path}/{ARCH}_model_weights_best.pth')
            print(f'Model saved as {ARCH}_model_weights_best.pth')
            best_eval_loss = eval_loss
    
    print("Best accuracy: {:.6f}".format(best_acc))

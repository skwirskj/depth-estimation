import torch
import numpy as np
import numpy.ma as ma
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from dataset import *
from torchvision import models
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
from unet import UNet
from torchsummaryX import summary
import time
from tqdm import tqdm
import pytorch_ssim
import math

def save_model_for_safety(model, epoch, optimizer, loss):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, "../depth_models/model_at_"+str(epoch)+".pt")

def save_model_for_inference(model):
    torch.save(model.state_dict(), '../depth_models/best_model.pt')

def plot_depth_map(dm, validity_mask):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)

    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))
    plt.colorbar()
    plt.show()

def train(trainloader, densemodel, criterion, optimizer, device, epoch):
    """
    Training function
    """
    start = time.time()
    running_loss = 0.0
    cnt = 0
    #print(trainloader.size())
    model = densemodel.train()
    for image in tqdm(trainloader):
        images = image[0].to(device)
        labels = image[1].to(device)
        #print("labels shape", labels.shape)
        # Downsampling label to match depth output
        labels = nn.functional.interpolate(labels.float(), size=(192, 256), mode='bilinear') 
        optimizer.zero_grad()
        output = model(images.float())
        loss = calc_loss(output, labels, 0.1, device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        cnt += 1
    end = time.time()
    running_loss /= cnt # avg loss across all batches

    if epoch % 10 == 0:
        save_model_for_safety(densemodel, epoch, optimizer, loss)

    print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
            (epoch, running_loss, end-start))
    return running_loss

def test(testloader, densemodel, criterion, device, epoch_lim, epoch, min_loss):
    losses = 0
    cnt = 0
    with torch.no_grad():
        densemodel = densemodel.eval()
        for image in tqdm(testloader):
            images = image[0].to(device)
            labels = image[1].to(device)
            labels = nn.functional.interpolate(labels.float(), size=(192, 256), mode='bilinear')

            output = densemodel(images.float())
            loss = calc_loss(output, labels, 0.1, device)
            losses += loss.item()
            cnt += 1
            if epoch == epoch_lim - 1:
                valid_mask = image[2].to(device)
                valid_mask = valid_mask.reshape((-1, 1, 384, 512))
                valid_mask = nn.functional.interpolate(valid_mask, size=(192, 256))
                plot_depth_map(torch.squeeze(output, dim=0).permute((1, 2, 0)).cpu().numpy(), torch.squeeze(valid_mask, dim=0).permute((1, 2, 0)).cpu().numpy())
    avg_loss = losses / cnt
    if (avg_loss < min_loss):
        min_loss = avg_loss
        save_model_for_inference(densemodel)
    print('\n', avg_loss)
    return avg_loss, min_loss 

def calc_loss(pred, true, lmda, device):
    return lmda*depth_loss(pred, true) + grad_loss(pred, true, device) + ssim_loss(pred, true)

# Assumes pred/true are size (batch_size, 1, 384, 512)
def grad_loss(pred, true, device):
    N, C, H, W = pred.shape
    # Compute x component of gradient
    sobel_x = torch.Tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
    sobel_x = sobel_x.view((1, 1, 3, 3))
    sobel_x = sobel_x.to(device)

    gx_pred = nn.functional.conv2d(pred, sobel_x, stride=1, padding=1)
    gx_true = nn.functional.conv2d(true, sobel_x, stride=1, padding=1)

    # Compute y component of gradient
    sobel_y = torch.Tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1,-2,-1]])
    sobel_y = sobel_y.view((1, 1, 3, 3))
    sobel_y = sobel_y.to(device)

    gy_pred = nn.functional.conv2d(pred, sobel_y, stride=1, padding=1)
    gy_true = nn.functional.conv2d(true, sobel_y, stride=1, padding=1)
    #plt.imshow(torch.squeeze(gx_true, dim=0).permute((1,2,0)).cpu())
    #plt.show()

    # Compute gradient loss as defined in paper
    loss = torch.sum(torch.abs(gx_true - gx_pred) + torch.abs(gy_true - gy_pred)) / (N * H * W) # Include N?

    return loss

def ssim_loss(pred, true):
    N, C, H, W = pred.shape
    # Returns scalar, so I assume it is computing for whole batch
    loss = (1 - pytorch_ssim.ssim(true, pred)) / 2

    return loss

def depth_loss(pred, true):
    N, C, H, W = pred.shape
    loss = torch.sum(torch.abs(true - pred)) / (N * H * W)

    return loss


# def eval(testloader, densemodel, device):
#   with torch.no_grad():
#     densemodel = densemodel.eval()
#     cnt = 0
    
#     for i, image in enumerate(testloader):
#       images = image[0].to(device)
#       truth = nn.functional.interpolate(image[1].float(), scale_factor=0.5, mode='bilinear').numpy()
#       pred = densemodel(images.float()).cpu().numpy()

#       n = pred.shape[0]*pred.shape[1]*pred.shape[2]*pred.shape[3]
#       rel = np.sum(abs(truth - pred) / truth) / n
#       rms = math.sqrt(np.sum((truth - pred)**2) / n)

#       valid_mask = image[2].to(device)
#       valid_mask = valid_mask.reshape((-1, 1, 768, 1024))
#       valid_mask = nn.functional.interpolate(valid_mask, scale_factor=0.5)
#       print(pred.shape)
#       plot_depth_map(pred[0, :,:,:].transpose((1, 2, 0)), torch.squeeze(valid_mask, dim=0).permute((1, 2, 0)).cpu().numpy())

def eval(testloader, densemodel, device):
  with torch.no_grad():
    densemodel = densemodel.eval()
    cnt = 0.0
    total_ap = 0.0
    preds = [[] for _ in range(len(testloader))]
    truths = [[] for _ in range(len(testloader))]
    accuracy = []
    for image in tqdm(testloader):
      images = image[0].to(device)
      pred = densemodel(images.float()).cpu().numpy().reshape(-1)
      truth = nn.functional.interpolate(image[1].float(), scale_factor=0.5, mode='bilinear').numpy().reshape(-1)
      masked_p = ma.masked_where(truth == 0, pred)
      masked_t = ma.masked_where(truth == 0, truth)
      #print(masked_t)
      res = 1.0 - (abs(masked_t-masked_p)/abs(masked_t))
      res[res < 0] = 0
      AP = np.mean(res)
      print('Image:', i , 'AP = ', np.round(AP*100, 3))
      accuracy.append(AP)

    return np.mean(accuracy)

def main():

    # Checking if CUDA is good to go
    device = None
    if torch.cuda.is_available():
        print("Using the GPU. You are good to go!")
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        raise Exception("WARNING: Could not find GPU! Using CPU only. \
    To enable GPU, please to go Edit > Notebook Settings > Hardware \
    Accelerator and select GPU.")

    # Can change this for your local machine
    high_level_path = "/media/lance/HDD/diode_data"
    train_diode = DIODE("new_meta.json", high_level_path, ('train'), ('indoors'))
    val_diode = DIODE("new_meta.json", high_level_path, ('val'), ('indoors'))
    #test_diode = DIODE("new_meta.json", high_level_path, ('test'), ('indoors'))
    print(len(train_diode))
    # Fixing the sizes for the train, validation, and test data
    # NOTE: These must add up to length of the data sets when splitting below
    train_size = 1
    val_size = 0
    test_size = 0

    # Splitting up the data into smaller subsets
    # The generator for selecting a random dataset is currently fixed just to be reproducible.
    #train_data, val_data, test_data, _ = torch.utils.data.random_split(val_diode, [train_size, val_size, test_size, 324], generator=torch.Generator().manual_seed(3))

    train_dl = DataLoader(train_diode, batch_size=8) # Trying to overfit
    val_dl = DataLoader(val_diode, batch_size=8)
    #test_dl = DataLoader(test_diode, batch_size=1) # 2 was highest for smaller densenet and resnet50
    # list        RGB batch        depth batch  depth mask batch
    # batch: [(8, 768, 1024, 3), (8, 768, 1024), (8, 768, 1024)]
    #for i, batch in enumerate(train_dl):
    #    plt.imshow(np.transpose(batch[0][0], (1, 2, 0)))
    #    plt.show()
    #    plot_depth_map(batch[1][0][0], batch[2][0])
    #    plt.show()

    densemodel = UNet().to(device)
    #net = Dense().to(device)
    
    learning_rate = 1e-3
    weight_decay = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(densemodel.parameters(), learning_rate, weight_decay=weight_decay, betas=[0.9, 0.999])
    num_epoch = 500 
    
    # visualizing the model
    #test = models.resnet50(pretrained=True)
    # print('Your network:')
    #newmodel = nn.Sequential(*(list(test.children())[:-2]))
    #for name in newmodel.children():
    #    print(name)
    #summary(densemodel, torch.rand((8,3,384,512)).to(device))
    

    #pdb.set_trace()
    #train model
    losses = []
    min_loss = 1e6
    for i in range(num_epoch):
        training_loss = train(train_dl, densemodel, criterion, optimizer, device, i)
        val_loss, min_loss = test(val_dl, densemodel, criterion, device, num_epoch, i, min_loss)  
        print('Validation loss: ', val_loss, "Min loss:", min_loss)
        if i%10 == 0:
          ap = eval(val_dl, densemodel, device)
          print('mAP: ', np.round(ap*100,3), '%')

        
    #mod = ResNet18().to(device)
    #mod.load_state_dict(torch.load('../depth_models/best_model.pt'))
    #eval(train_dl, mod, device)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

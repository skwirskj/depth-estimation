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
from resnet18 import Dense 
from torchsummaryX import summary
import time
from tqdm import tqdm
import pytorch_ssim
import math
import os
import itertools

def save_model_for_safety(model, epoch, optimizer, loss, path):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, "../model_at_"+str(epoch)+".pt")

def save_model_for_inference(model, path):
    torch.save(model.state_dict(), '../best_model.pt')

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

def make_training_plot(name="CNN Training"):
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    plt.suptitle(name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AP")
    return axes

def update_training_plot(axes, epoch, stats):
    splits = ["Train", "Validation"]
    metrics = ["Loss", "AP"]
    colors = ["r", "b"]
    for i, metric in enumerate(metrics):
        for j, split in enumerate(splits):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            axes[i].plot(
                range(epoch - len(stats) + 1, epoch + 1),
                [stat[idx] for stat in stats],
                linestyle="--",
                marker="o",
                color=colors[j],
            )
        axes[i].legend(splits[: int(len(stats[-1]) / len(metrics))])
    plt.pause(0.00001)

def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats

def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

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
        #print(images.shape)
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
    return running_loss

def average_relative_error(y_pred, y_true):
    """
    Perform the threshold accuracy measurement on a single batch
    Inputs: y_true, true depth map size: [batch, 1, H, W]
            y_pred, the predicted output from the model size: [batch, 1, H, W]
    output: The average relative error calculation performed on each image thencount   averaged accross the entire batch
    """
    N = (y_pred.shape[2] * y_pred.shape[3]) # H x W
    return torch.mean(torch.sum(torch.abs(y_true - y_pred), axis=(1,2,3))) / N
    
def root_mean_squared_error(y_pred, y_true):
    """
    Perform the threshold accuracy measurement on a single batch
    Inputs: y_true, true depth map size: [batch, 1, H, W]
            y_pred, the predicted output from the model size: [batch, 1, H, W]
    output: The root mean squared error calculation performed on each image then
            averaged accross the entire batch
    """
    N = (y_pred.shape[2] * y_pred.shape[3]) # H x W
    return torch.mean(torch.sqrt(torch.sum((y_true - y_pred) ** 2, axis=(1,2,3)) / N))

def average_log_10(y_pred, y_true):
    """
    Perform the threshold accuracy measurement on a single batch
    Inputs: y_true, true depth map size: [batch, 1, H, W]
            y_pred, the predicted output from the model size: [batch, 1, H, W]
    output: The average log10 error calculation performed on each image then
            averaged accross the entire batch
    """
    N = (y_pred.shape[2] * y_pred.shape[3]) # H x W
    return torch.mean(torch.sum(torch.abs(torch.log10(y_true) - torch.log10(y_pred)), axis=(1,2,3)) / N)

def threshold_accuracy(y_true, y_pred, thr):
    """
    Perform the threshold accuracy measurement on a single batch
    Inputs: y_true, true depth count[batch, 1, H, W]
            y_pred, the predicted output from the model size: [batch, 1, H, W]
            thr: the threshold value corresponding to max(y/y', y'/y) < thr
    output: The average pixel count that satisfy max(y/y', y'/y) < thr
    """
    # still needs to be vectorized to improve the performance!
    count = torch.zeros(y_true.shape)
    for batch in range(y_true.shape[0]):
        for i in range(y_true.shape[2]):
            for j in range(y_true.shape[3]):
                max_pixel = max(y_true[batch,:,i,j] / y_pred[batch,:,i,j], y_pred[batch,:,i,j] / y_true[batch,:,i,j])
                if max_pixel < thr:
                    count[batch,:,i,j] = 1
    return torch.mean(torch.sum(count, axis=(1,2,3)))


def evaluate_epoch(loader, model, device):
    """
    Perform the threshold accuracy measurement on a single batch
    Inputs: loader, the list that contains each input batch and the associated batch of labels.
            device, either CPU or GPU.
    output: The average of each evaluation calculation accross each image in the loader set.
    """
    ave = []
    rmse = []
    avlog10 = []
    tresh_acc = []
    for image in loader: # evaluate each batch
        with torch.no_grad():
            X = image[0].to(device)
            y = image[1].to(device)
            #X = X[:,0,:,:,]
            #print(X.shape)
            #print("labels shape", labels.shape)
            # Downsampling label to match depth output
            y = nn.functional.interpolate(y.float(), size=(192, 256), mode='bilinear') 
            output = model(X)
            ave.append(average_relative_error(y, y))
            rmse.append(root_mean_squared_error(y, y))
            avlog10.append(average_log_10(y, y))
            #tresh_acc  = []
        #for thr in [1.25, 1.25 **2, 1.25 ** 3]:
            tresh_acc.append(threshold_accuracy(y, y, 1.25))
    return np.mean(ave), np.mean(rmse), np.mean(avlog10), np.mean(thresh_acc)

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
            #if epoch == epoch_lim - 1:
            #    valid_mask = image[2].to(device)
            #    valid_mask = valid_mask.reshape((-1, 1, 384, 512))
            #    valid_mask = nn.functional.interpolate(valid_mask, size=(192, 256))
            #    plot_depth_map(torch.squeeze(output, dim=0).permute((1, 2, 0)).cpu().numpy(), torch.squeeze(valid_mask, dim=0).permute((1, 2, 0)).cpu().numpy())
    avg_loss = losses / cnt
    #if (avg_loss < min_loss):
    #    min_loss = avg_loss
    #    save_model_for_inference(densemodel)
    #print('\n', avg_loss)
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

    
def eval(testloader, densemodel, device):
  with torch.no_grad():
    densemodel = densemodel.eval()
    cnt = 0.0
    total_ap = 0.0
    preds = [[] for _ in range(len(testloader))]
    truths = [[] for _ in range(len(testloader))]
    accuracy = []
    for i, image in enumerate(tqdm(testloader)):
      images = image[0].to(device)
      pred = densemodel(images.float()).cpu().numpy().reshape(-1)
      truth = nn.functional.interpolate(image[1].float(), scale_factor=0.5, mode='bilinear').numpy().reshape(-1)
      masked_p = ma.masked_where(truth == 0, pred)
      masked_t = ma.masked_where(truth == 0, truth)
      #print(masked_t)
      res = 1.0 - (abs(masked_t-masked_p)/masked_t)
      res[res < 0] = 0
      AP = np.mean(res)
      print('Image:', i , 'AP = ', np.round(AP*100, 3))
      accuracy.append(AP)

    return np.mean(accuracy)

def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)

def main():

    # Checking if CUDA is good to go
    device = None
    if torch.cuda.is_available():
        print("Using the GPU. You are good to go!")
    else:
        device = torch.device('cpu')
        raise Exception("WARNING: Could not find GPU! Using CPU only. \
    To enable GPU, please to go Edit > Notebook Settings > Hardware \
    Accelerator and select GPU.")

    # Can change this for your local machine
    high_level_path = "D:\\School Tom\\EECS 442\\442-depth-estimation-main\\diode_data"
    train_diode = DIODE("new_meta.json", high_level_path, ('train'), ('indoors'))
    val_diode = DIODE("new_meta.json", high_level_path, ('val'), ('indoors'))
    test_diode = DIODE("new_meta.json", high_level_path, ('test'), ('indoors'))
    print(len(train_diode))
    print(len(val_diode))
    print(len(test_diode))
    # Fixing the sizes for the train, validation, and test data
    # NOTE: These must add up to length of the data sets when splitting below
    #train_size = 8
    #val_size = 8
    #test_size = 8

    # Splitting up the data into smaller subsets
    # The generator for selecting a random dataset is currently fixed just to be reproducible.
    #train_data, val_data, test_data, _ = torch.utils.data.random_split(val_diode, [train_size, val_size, test_size, 219], generator=torch.Generator().manual_seed(3))

    train_dl = DataLoader(train_diode, batch_size=32) # Trying to overfit
    print(len(train_dl))
    val_dl = DataLoader(train_diode, batch_size=32)
    test_dl = DataLoader(train_diode, batch_size=32) # 2 was highest for smaller densenet and resnet50
    # list        RGB batch        depth batch  depth mask batch
    # batch: [(8, 768, 1024, 3), (8, 768, 1024), (8, 768, 1024)]
    # for i, batch in enumerate(train_dl):
    #     plt.imshow(np.transpose(batch[0][0], (1, 2, 0)))
    #     plt.show()
    #     plot_depth_map(batch[1][0][0], batch[2][0])
    #     plt.show()

    densemodel = Dense().to(device)
    #net = Dense().to(device)
    
    learning_rate = 1e-4
    weight_decay = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(densemodel.parameters(), learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    num_epoch = 10
    
    # visualizing the model
    #test = models.resnet50(pretrained=True)
    # print('Your network:')
    #newmodel = nn.Sequential(*(list(test.children())[:-2]))
    #for name in newmodel.children():
    #    print(name)
    #summary(newmodel, torch.rand((1,3,1024,768)))

    #pdb.set_trace()
    #train model
    axes = make_training_plot()
    stats = []
    min_loss = 1e6
    model, start_epoch, stats = restore_checkpoint(densemodel, high_level_path)
    epoch = start_epoch
    while epoch < num_epoch:
        print(train_dl)
        training_loss = train(train_dl, densemodel, criterion, optimizer, device, epoch)
        print("Training loss: ", training_loss)
        # evaluate the metrics and graph them
        val_loss, min_loss = test(val_dl, densemodel, criterion, device, num_epoch, epoch, min_loss)
        print("Validation loss: ", val_loss)
        train_ap = eval(train_dl, densemodel, device)
        val_ap = eval(val_dl, densemodel, device)

        #print('mAP: ', np.round(ap*100,3), '%')
        epoch_stats = [training_loss, train_ap, val_loss, val_ap]
        #####################
        #train_are, train_rmse, train_log, train_thesh = evaluate_epoch(train_dl, device)
        #val_are, val_rmse, val_log, val_thesh = evaluate_epoch(val_dl, device)
        #epoch_stats = [
        #    val_are,
        #    val_rmse,
        #    val_log,
        #    val_thesh,
        #    train_are,
        #    train_rmse,
        #   train_log,
        #    train_thesh,
        #]
        stats.append(epoch_stats)
        update_training_plot(axes, epoch, stats)
        #val_loss = test(val_dl, densemodel, criterion, device, num_epoch, epoch)
        #####################
        save_checkpoint(densemodel, epoch, high_level_path, stats)


    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import importlib
import torch.nn.functional as F
import torch.optim as optim



from model import GazeSwinTransformer, HybridLoss

if __name__ == "__main__":
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    readername = config["reader"]
    dataloader = importlib.import_module("reader." + readername)

    config = config["train"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["save"]["model_name"]

    folder = os.listdir(labelpath)
    folder.sort()

    i = int(sys.argv[2])
    if i in list(range(15)):
        trains = copy.deepcopy(folder)
        tests = trains.pop(i)
        print(f"Train Set:{trains}")
        print(f"Test Set:{tests}")

    trainlabelpath = [os.path.join(labelpath, j) for j in trains]

    savepath = os.path.join(config["save"]["save_path"], f"checkpoint/{tests}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Read data")
    dataset = dataloader.txtload(trainlabelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=0, header=True)

    print("Model building")
    net = GazeSwinTransformer()
    net.train()
    net.to(device)

    print("optimizer building")
    loss_op = HybridLoss().cuda()
    base_lr = config["params"]["lr"]
    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.AdamW(net.parameters(), lr=base_lr, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    


print("Training")
length = len(dataset)
total = length * config["params"]["epoch"]
cur = 0
timebegin = time.time()
best_loss = float('inf')

with open(os.path.join(savepath, "train_log"), 'w') as outfile:
    for epoch in range(1, config["params"]["epoch"]+1):
        epoch_loss = 0.0  
        for i, (data, label) in enumerate(dataset):
            data["face"] = F.interpolate(data["face"], size=(224, 224), mode='bilinear', align_corners=False).to(device)
            label = label.to(device)
            gaze, gaze_bias = net(data)
            loss = loss_op(gaze, label, gaze_bias)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cur += 1
            
            if i % 20 == 0:  
                timeend = time.time()
                resttime = (timeend - timebegin) / cur * (total - cur) / 3600
                log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()
                outfile.flush()
            
        epoch_loss = epoch_loss / len(dataset)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(net.state_dict(), os.path.join(savepath, f"Best_{epoch}_{modelname}.pt"))

        if epoch % config["save"]["step"] == 0:  
            torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

import model  # gaze 360 train
import numpy as np
import importlib
import torch
import torch.nn as nn
import time
import sys
import os
import yaml
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from model import GazeSwinTransformer, HybridLoss

if __name__ == "__main__":
    config = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)
    readername = config["reader"]
    dataloader = importlib.import_module("reader." + readername)

    config = config["train"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["save"]["model_name"]

    savepath = os.path.join(config["save"]["save_path"], f"checkpoint")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Read data")
    dataset = dataloader.txtload(labelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=4, header=True)

    print("Model building")
    net = model.GazeSwinTransformer()
    net.to(device)

    # 
    checkpoint_path = os.path.join(savepath, "modelsave/checkpoint/Iter_60_Gaze360.pt")
    if os.path.exists(checkpoint_path):
        net.load_state_dict(torch.load(checkpoint_path))
    
    net.train()

    print("optimizer building")
    loss_op = model.HybridLoss().cuda()
    base_lr = config["params"]["lr"]

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Training")
    length = len(dataset)
    total = length * config["params"]["epoch"]
    cur = 0
    timebegin = time.time()
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        for epoch in range(0, config["params"]["epoch"] + 1):  
            for i, (data, label) in enumerate(dataset):
                data["face"] = data["face"].to(device)
                label = label.to(device)

                transform = transforms.Compose([
                    transforms.Resize((224, 224))
                ])

                data["face"] = transform(data["face"])

                gaze, gaze_bias = net(data)

                loss = loss_op(gaze, label, gaze_bias)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                scheduler.step()
                cur += 1

                if i % 20 == 0:
                    timeend = time.time()
                    resttime = (timeend - timebegin) / cur * (total - cur) / 3600
                    log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()

                if epoch % config["save"]["step"] == 0:
                    torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

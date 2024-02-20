import model
import importlib
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
import torch.nn.functional as F


def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

if __name__ == "__main__":
  config = yaml.load(open(sys.argv[1]), Loader = yaml.FullLoader)
  readername = config["reader"]
  dataloader = importlib.import_module("reader." + readername)

  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  folder = os.listdir(labelpath)
  folder.sort()
  i = int(sys.argv[2])

  if i in range(15): 
    tests = folder[i]
    print(f"Test Set: {tests}")
   
    savepath = os.path.join(loadpath, f"checkpoint/{tests}")
    
    if not os.path.exists(os.path.join(loadpath, f"evaluation/{tests}")):
      os.makedirs(os.path.join(loadpath, f"evaluation/{tests}"))

    print("Read data")
    dataset = dataloader.txtload(os.path.join(labelpath, tests), imagepath, 32, num_workers=4, header=True, shuffle=False)

    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]
    
    best_avg = float("inf")  
    best_model_iter = None

    for saveiter in range(begin, end+step, step):
      print("Model building")
      net = model.GazeSwinTransformer()
      statedict = torch.load(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"))
      statedict = torch.load(os.path.join(savepath, f"Best_{saveiter}_{modelname}.pt"))

      net.to(device)
      net.load_state_dict(statedict)
      net.eval()

    

        
      print(f"Test {saveiter}")
      length = len(dataset)
      accs = 0
      count = 0
      with torch.no_grad():
        with open(os.path.join(loadpath, f"evaluation/{tests}/{saveiter}.log"), 'w') as outfile:
          outfile.write("name results gts\n")
          for j, (data, label) in enumerate(dataset):
            img = data["face"].to(device)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            names = data["name"]


            img = {"face":img}
            gts = label.to(device)
             
            gazes, gaze_biases = net(img)
            for k, gaze in enumerate(gazes):
              gaze = gaze.cpu().detach().numpy()
              count += 1
              accs += angular(gazeto3d(gaze), gazeto3d(gts.cpu().numpy()[k]))
              
              name = [names[k]]
              gaze = [str(u) for u in gaze] 
              gt = [str(u) for u in gts.cpu().numpy()[k]] 
              log = name + [",".join(gaze)] + [",".join(gt)]
              outfile.write(" ".join(log) + "\n")

        
          loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
          outfile.write(loger)
          print(loger)
            
          if accs/count < best_avg:
              best_avg = accs/count
              best_model_iter = saveiter
    print(f"The best performing model is Best_{best_model_iter}_{modelname}.pt with avg: {best_avg}")
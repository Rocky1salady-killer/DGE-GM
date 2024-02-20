# DGE-GM
Appearance-based Driver 3D Gaze Estimation Using GRM and Mixed Loss Strategies

### The project code is currently being organized and released incrementally.


Inference results on the dmd dataset:
<p align="center">
  <img src="inference/inference results on dmd dataset.gif" alt="animated" />
</p>



## Introduction
Two projects were provided for leave-one-person-out evaluation and the evaluation of common training-test split.
They have the same architecture but different `train.py` and `test.py`.

Each project contains following files/folders.
- `model.py`, the model code.
- `train.py`, the entry for training.
- `test.py`, the entry for testing.
- `config/`, this folder contains the config of experiments for each dataset. To run our code, **you should write your own** `config.yaml`. 
- `reader/`, the data loader code. You can use the provided reader or write your own reader.

## Getting Started


For train, you should change:
1. `train.save.save_path`, The model is saved in the `$save_path$/checkpoint/`.
2. `train.data.image`, This is the path of image, please use the provided data processing code.
3. `train.data.label`, This is the path of label.
4. `reader`, This indicates the used reader. It is the filename in `reader` folder, e.g., *reader/reader_mpii.py* ==> `reader: reader_mpii`.

For test, you should change:
1. `test.load.load_path`, it is usually the same as `train.save.save_path`. The test result is saved in `$load_path$/evaluation/`.
2. `test.data.image`, it is usually the same as `train.data.image`.
3. `test.data.label`, it is usually the same as `train.data.label`.

### Data preprocessing
Data preprocessing is for step 2 of the Getting Started training.

1.On the preprocessing of the MPIIFaceGaze dataset
The [MPIIFaceGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation) can be downloaded.
The code contains following parameters：
```
root = "/home/cyh/dataset/Original/MPIIFaceGaze"
sample_root = "/home/cyh/dataset/Original/MPIIGaze/Origin/Evaluation Subset/sample list for eye image"
out_root = "/home/cyh/dataset/EyeBased/MPIIGaze"
```
The root is the path of MPIIFaceGaze.
The sample_root indicates the sample list in MPIIGaze. Note that, this file is not contained in MPIIFaceGaze. You should download MPIIGaze for this file.
The out_root is the path for saving result.
To use the code, you should set the three parameters first., and run：
```
cd data processing
python data_processing_mpii.py
```

2.On the preprocessing of the Gaze360 dataset
The [Gaze360 dataset](https://orion.hyper.ai/tracker/download?torrent=20170) can be downloaded.

The code contains following parameters:
```
root = "/home/cyh/dataset/Original/Gaze360/"
out_root = "/home/cyh/dataset/FaceBased/Gaze360"
```
The root is the path of original Gaze360 dataset.
The out_root is the path for saving result file.
To use the code, you should first set the two paramters, and run
```
cd data processing
python data_processing_gaze360.py
```



#### We are grateful to  [GazeHub@Phi-ai Lab](https://phi-ai.buaa.edu.cn/Gazehub/)  for their contributions to this part of the work.

### Installation

The requirements are listed in the `requirement.txt` file. To create your own environment, an example is:
```
pip install -r requirements.txt
```
### Train
Training on the MPIIFaceGaze dataset, you can run in the leaveout folder：
```
cd MPIIFaceGaze/Leaveout
python train.py config/config_mpii.yaml 0
```
or
```
bash run.sh train.py config/config_mpii.yaml
```
Training on Gaze 360 dataset, you can run in the traintest folder:
```
cd Gaze360/Traintest
python train.py config/config_mpii.yaml
```

### Test
Testing on the MPIIFaceGaze dataset, you can run in the leaveout folder：：
```
cd MPIIFaceGaze/Leaveout
python test.py config/config_mpii.yaml 0
```
or
```
bash run.sh test.py config/config_mpii.yaml
```
Testing on Gaze 360 dataset, you can run in the traintest folder:
```
python test.py config/config_mpii.yaml
```
### Inference
For inference videos or images, please run:
```
cd inference
python inference-1.py
```
For inference using the local camera, please run:
```
cd inference
python inference-2.py
```


### Result
After training or test, you can find the result from the `save_path` in `config_mpii.yaml`. 

## Acknowledgments

Our work is based on Swin transformer, Gaze360 and Fullface.  We appreciate the previous open-source repository [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Gaze360](https://github.com/yihuacheng/Gaze360) and [Fullface](https://github.com/yihuacheng/Full-face).

Please follow their outstanding work:

```
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={10012--10022},
  year={2021}
}

@article{Cheng2021Survey,
        title={Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark},
        author={Yihua Cheng and Haofei Wang and Yiwei Bao and Feng Lu},
        journal={arXiv preprint arXiv:2104.12668},
        year={2021}
}

@InProceedings{Kellnhofer_2019_ICCV,
	author = {Kellnhofer, Petr and Recasens, Adria and Stent, Simon and Matusik, Wojciech and Torralba, Antonio},
	title = {Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
	booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	month = {October},
	year = {2019}
}

@inproceedings{Zhang_2017_CVPRW,
	title={It’s written all over your face: Full-face appearance-based gaze estimation},
	author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
	booktitle={The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
	pages={2299--2308},
	month={July},
	year={2017},
	organization={IEEE}
}
```




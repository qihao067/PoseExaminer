# PoseExaminer (CVPR2023)

<img src="fig/fig1.jpg" alt="fig1" style="zoom:20%;" />

#### [PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation](https://arxiv.org/pdf/2303.07337.pdf) 
#### CVPR2023
#### [Qihao Liu](https://qihao067.github.io/) | [Adam Kortylewski](https://gvrl.mpi-inf.mpg.de/) | [Alan Yuille](https://cogsci.jhu.edu/directory/alan-yuille/) 

This repository contains the code and model of [PoseExaminer](https://arxiv.org/pdf/2303.07337.pdf). It is built on [MMHuman3D](https://github.com/open-mmlab/mmhuman3d) to be compatible with various human pose and shape (HPS) estimation methods. By following the instructions provided, you will be able to run our adversarial examiner on any HPS method (e.g. [PARE](https://pare.is.tue.mpg.de/)). Then, you can improve the model's performance by fine-tuning it on the failure modes (e.g., the released code improves PARE from 81.81 to 73.65 MPJPE on the 3DPW dataset). More coming soon.

______



## Requirements

The code has been tested with PyTorch 1.7.1 and Cuda 11.4.

```
conda env create -f adv_env.yaml

mkdir third_party
cd third_party

pip install git+https://github.com/mkocabas/yolov3-pytorch.git
pip install git+https://github.com/mkocabas/multi-person-tracker.git
pip install git+https://github.com/giacaglia/pytube.git

git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
python setup.py develop

pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.1.1/index.html
pip install "mmdet<=2.25.1"
pip install "mmpose<=0.28.1"

cd ../..
```

________



## Data & Model Preparation

1. Download the original pertained [PARE model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/without_mosh/hrnet_w32_conv_pare.pth?versionId=CAEQOhiBgMCi4YbVgxgiIDgzYzFhMWNlNDE2NTQwN2ZiOTQ1ZGJmYTM4OTNmYWY5) to this path [`PoseExaminer/Methods/PARE/master/`](https://github.com/qihao067/PoseExaminer/tree/main/Methods/PARE/master)

2. Download the human body model and other resources following this [page](https://github.com/open-mmlab/mmhuman3d/tree/main/configs/pare) in MMHuman3D, and put them to this path: [`PoseExaminer/data`](https://github.com/qihao067/PoseExaminer/tree/main/data)

3. Download the human body model used for [VPoser](https://github.com/nghorbani/human_body_prior/) from this [page](https://smpl-x.is.tue.mpg.de/), and put them to this path: [`PoseExaminer/utils_PoseExaminer/support_data/dowloads`](PoseExaminer/utils_PoseExaminer/support_data/dowloads)

4. Prepare the training data for PARE following this [page](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md) in MMHuman3D. You only need to download the data for PARE to run the code in this repo. If you only want to evaluate our model, you can just download the 3DPW dataset.

   

   ##### Finally, you should have the following file structure to run our code:

   ```
   PoseExaminer   
       ├── configs
       ├── data
       │   ├── body_models                                   # should be the same as in MMHuman3D
       │   ├── preprocessed_datasets                         # should be the same as in MMHuman3D
       │   ├── pretrained_models                             # should be the same as in MMHuman3D
       │   ├── datasets                                      # should be the same as in MMHuman3D
       │   │   ├── coco
       │   │   ├── h36m
       │   │   └── ...
       │   └── gmm_08.pkl
       ├── Methods
       │   └── PARE
       │       ├── master
       │       │   └── hrnet_w32_conv_pare.pth               # The original PARE checkpoint
       │       └── ours
       │           └── exp_best_pare.pth                     # The checkpoint after training with PoseExaminer. It is only needed to run `test.sh`
       ├── mmhuman3d
       ├── third_party
       ├── tools
       ├── utils_PoseExaminer
       │   ├── data_generated
       │   ├── models
       │   └── support_data
       │       ├── uv_bk_map
       │       └── dowloads                                  # All can be downloaded from VPoser
       │           ├── models
       │           │   ├── smpl
       │           │   │   ├── basicmodel_m_lbs_10_207_0_v1.0.0_lqh.npz
       │           │   │   └── basicmodel_m_lbs_10_207_0_v1.0.0.pkl
       │           │   └── smplx
       │           │       ├── female
       │           │       ├── male
       │           │       └── neutral
       │           ├── vposer_v2_05
       │           └── amass_sample.npz
       ├── train.sh
       └── test.sh
   ```

____________



## Evaluation

You can evaluate the model we provide using `test.sh`:

```
bash test.sh
```

We released our model here. Please download the model and put it to this path ([`PoseExaminer/Methods/PARE/ours`](PoseExaminer/Methods/PARE/ours)) to run evaluation. The models are tested on the 3DPW dataset.

|                            | MPJPE | PA-MPJPE | PVE    | Download                                                     |
| -------------------------- | ----- | -------- | ------ | ------------------------------------------------------------ |
| PARE (Original)            | 81.81 | 50.78    | 102.27 | [Model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/pare/without_mosh/hrnet_w32_conv_pare.pth?versionId=CAEQOhiBgMCi4YbVgxgiIDgzYzFhMWNlNDE2NTQwN2ZiOTQ1ZGJmYTM4OTNmYWY5) |
| PARE + PoseExaminer (Ours) | 73.65 | 47.25    | 91.29  | [Model](https://drive.google.com/drive/folders/1gRTey3_TpwjMV1HVzNKamjgcv643kLFh?usp=sharing) |

______



## Adversarial Examination and Training with PoseExaminer

By running the `train.sh`, you will first run the PoseExmainer on PARE to search for the failure modes with 40 agents, and then fine-tune the PARE mode with the discovered failure cases (i.e., reproduce the results we reported). We repeat the entire process for 5 times. All intermediate results and checkpoints will be saved in `ROOTNAME` (see `train.sh`). 

```
bash train.sh
```

The `train.sh` file will do the following five things:

1. Run PoseExaminer on a given model (Adversarial examination on current model)
2. Sample 40*500 failure cases for training
3. Train the model (Train PARE)
4. Test the trained model on 3DPW
5. Save all intermetiate results (checkpoints, failure modes, generated training images, results, etc).

______



## Notes

1. Our code release exclusively focuses on the articulated pose. By following the instructions, you should be able to conduct a comprehensive evaluation of the SOTA method (i.e., PARE) and enhance its performance on the 3DPW dataset from the reported 81.81 to 73.65, following approximately 10 hours of training. This improvement slightly surpasses the number reported in our CVPR paper (74.44 MPJPE).

2. 
   Since the synthetic training data depends on the failure modes identified by the PoseExaminer, and we disable the loss function for policy distance, the results may vary to some degree. When releasing the code, we conducted the experiment ten times, and we achieved an average MPJPE of 75.40 on the 3DPW dataset. If the obtained result falls short of your expectations, you can simply rerun the experiment

3. Our experiment required 8 GPUs with a minimum RAM capacity of 24 GB to employ 40 agents at the same time. However, to run the experiment on a reduced number of GPUs, it is possible to employ a smaller number of agents during each adversarial search process and repeat the process multiple times. As an example, you could run the search process four times, utilizing 10 agents during each process (leading to 40 failure modes in total).  To support such experiments, we have disabled the loss function associated with policy distance. Nevertheless, if you have enough GPUs, you can add this loss function back.

4. Please note that the poses may appear quite unusual. This is because our search for articulated poses is conducted under very simple conditions. We do not consider any other challenges such as occlusion, camera viewpoint, OOD body shapes, and we utilize a white background and easily distinguishable human textures during evaluation. (Different parts of the body, e.g., the torso, the thighs, the calves, and the arms, have different textures). The current SOTA methods demonstrate high accuracy in such easy settings, and as a result, only the unusual poses tend to result in significant errors.

   In addition, our objective is to identify the worst cases that cause the largest errors (e.g. an MPJPE exceeding 300). The only constraint we impose on the poses is their physical plausibility. It is evident that the most unusual poses typically lead to the largest errors. If your want to find poses that cause relatively large errors but still look normal, we suggest you (1) use a more stringent physical constraint by e.g. setting narrower joint ranges for each human joint, (2) stop the search process once the error surpasses a predefined threshold (e.g., 90 or 120), rather than continuing to search for the worst poses that have very large MPJPE.

   We also find these strange/less common poses useful to improve the performance of current methods.

____________



## License

The code in this repository is released under the MIT License. [MMHuman3D](https://github.com/open-mmlab/mmhuman3d) is released under the [Apache 2.0 license](https://github.com/open-mmlab/mmhuman3d/blob/main/LICENSE). Some supported methods may carry [additional licenses](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/additional_licenses.md).

_____________



## BibTeX

```
@inproceedings{liu2023poseexaminer,
  title={PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation},
  author={Liu, Qihao and Kortylewski, Adam and Yuille, Alan L},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={672--681},
  year={2023}
}
```


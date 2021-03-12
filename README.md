## ESNet: An  Efficient  Stereo  Matching  Network

Code in PyTorch for paper "ES-Net:  An  Efficient  Stereo  Matching  Network" submitted to IROS 2021 [[Paper Link]](https://arxiv.org/abs/2103.03922). 

## Dependency
Python 3.6

PyTorch(1.2.0+)

torchvision 0.2.0

## Usage
"networks/ESNet.py" and "networks/ESNet_M.py" contains the implementation of the proposed efficient stereo matching network.

To train the ESNet on SceneFlow dataset, you will need to modify the path to the dataset in the "exp_configs/esnet_sceneflow.conf". Then run
```
dnn=esnet_sceneflow bash train.sh
```

## Citation
If you find the code and paper is useful in your work, please cite
```
@misc{huang2021esnet,
    title={ES-Net: An Efficient Stereo Matching Network},
    author={Zhengyu Huang and Theodore B. Norris and Panqu Wang},
    year={2021},
    eprint={2103.03922},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

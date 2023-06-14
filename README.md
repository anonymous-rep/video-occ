# CTx
Code for CTx-Net for vedio actiton recognition under occlusion

### Installation

The code uses **Python 3.9** and it is tested on PyTorch GPU version 1.11, with CUDA-11.6

### Setup CTx-Net Virtual Environment

```
virtualenv --no-site-packages <your_home_dir>/.virtualenvs/CTx
source <your_home_dir>/.virtualenvs/CTx/bin/activate
```

### Clone the project and install requirements

```
git clone https://github.com/shroglck/CTx.git
cd CTx
pip install -r requirements.txt
```

# 



 

#### Evaluate the classification performance of a model

Run the following command in the terminal to evaluate a model on the full test dataset:
```
python compose_tester.py 
```


## Initializing CTx-net Parameters

CTx-Net parameters (vMF kernels and mixture models ) are initialized by clustering the feature vectors

```
python vc_cluster_fine.py
``` 

Furthermore, we initialize the mixture models by EM-type learning.
The initial cluster assignment for the EM-type learning is computed based on the similarity of the vMF encodings of the training images.
To compute the similarity matrices use:
 
```
python simmat_finer.py
``` 

Similarly the mixture model weights can be learned using
```
python mix_model_lear_finer.py
```
To train the model
```
python compose_trainer_2.py
```


## Acknowledgement 

This code has been adapted from
```
@inproceedings{CompNet:CVPR:2020,
  title = {Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion},
  author = {Kortylewski, Adam and He, Ju and Liu, Qing and and Yuille, Alan},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2020},
  month_numeric = {6}
}

@article{kortylewski2021compositional,
  title={Compositional convolutional neural networks: A robust and interpretable model for object recognition under occlusion},
  author={Kortylewski, Adam and Liu, Qing and Wang, Angtian and Sun, Yihong and Yuille, Alan},
  journal={International Journal of Computer Vision},
  volume={129},
  number={3},
  pages={736--760},
  year={2021},
  publisher={Springer}
}

```


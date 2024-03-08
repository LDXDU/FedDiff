# FedDiff_Under review by TCSVT
This paper is currently under review by IEEE TCSVT, and the diffusion framework of the FedDiff algorithm part will be disclosed.
Implementation of the vanilla federated learning paper : https://arxiv.org/abs/2401.02433

Experiments are produced on：Houston2013, Trento, The MUUFL dataset (both non-IID)).
![framework](https://github.com/LDXDU/FedDiff/assets/68802236/df30d339-c702-4c37-b647-538f9f5fa6d5)

## Requirments
- Python 3.8
- Pytorch
- Torchvision
- Numpy
- Matplotlib
- Scipy
- Tqdm

## Data
In terms of data, we respectfully refer to Hong's processing methods, such as splitting the image into 7*7 dimensions (subject to U-net restrictions in the later period, reshape will be carried out), and normalization in a strip way (0-1), but for the protection of data copyright, we  respectfully ask you to go to the link to download the data set and make a reference statement:https://github.com/danfenghong/IEEE_TGRS_MDL-RS

## Running the experiments
This project is divided into two parts at the algorithm level, namely the fusion part and the classification part:
- To run the diffusion model:
  
```powershell
python train_unet.py 
```

- To run the classifer model (The code will be published after acceptance of the paper):

```powershell
python train_classifier.py
```

```powershell
python eval_classifier.py
```

## Cite
If you are interested in my work, please refer to our work when using or referring, thank you very much!

```
Li D X, Xie W, Wang Z X, et al. FedDiff: Diffusion Model Driven Federated Learning for Multi-Modal and Multi-Clients[J]. arXiv preprint arXiv:2401.02433, 2023.
```
```
@article{li2023feddiff,
  title={FedDiff: Diffusion Model Driven Federated Learning for Multi-Modal and Multi-Clients},
  author={Li, DaiXun and Xie, Weiying and Wang, ZiXuan and Lu, YiBing and Li, Yunsong and Fang, Leyuan},
  journal={arXiv preprint arXiv:2401.02433},
  year={2023}
}
```


<h1 align="center">mann-pytorch</h1>

<p align="center">
   <a href="https://github.com/ami-iit/bipedal-locomotion-framework/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-BSD_3--Clause-orange.svg" alt="Size" class="center"/></a>
</p>

The **mann-pytorch** project is a PyTorch implementation of the Mode-Adaptive Neural Networks (MANN) architecture, originally proposed in H. Zhang, S. Starke, T. Komura, and J. Saito, “Mode-adaptive neural
networks for quadruped motion control,” ACM Trans. Graph., vol. 37,
no. 4, pp. 1–11, 2018.

---

<p align="center">
  <b>⚠️ REPOSITORY UNDER DEVELOPMENT ⚠️</b>
  <br>We cannot guarantee stable API
</p>

---

## Installation

Install `python3` and `pip` via:

```bash
sudo apt-get install python3.8 python3-pip
```

Clone and install the repo:

```bash
git clone https://github.com/ami-iit/mann-pytorch.git
cd mann-pytorch
pip install .
```

## Usage

### Training

You can execute a sample training script by:

```bash
cd mann-pytorch/scripts
python3 training.py
```

The training data will be periodically stored in a dedicated `mann-pytorch/models/storage_<training_start_time>` folder. You can also monitor the training progress by:

```bash
cd mann-pytorch/models/storage_<training_start_time>
python3 -m tensorboard.main --logdir=logs
```

### Testing

You can execute a sample testing script by:

```bash
cd mann-pytorch/scripts
python3 testing.py
```

The average loss of the learned model on the testing dataset will be printed. Moreover, you will be able to inspect the learned model performances by comparing the ground truth and the predicted output on each instance of the testing dataset. 

## Maintainer

This repository is maintained by:

| |                                                        |
|:---:|:------------------------------------------------------:|
| [<img src="https://user-images.githubusercontent.com/41757826/114039258-e334f080-9882-11eb-8037-ac7341666d21.png" width="40">](https://github.com/GitHubUserName) | [@paolo-viceconte](https://github.com/paolo-viceconte) |

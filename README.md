# HashGAN：基于PC-WGAN的深度哈希

这是CVPR 2018论文 ["HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_HashGAN_Deep_Learning_CVPR_2018_paper.pdf) 的复现仓库。论文原仓库指路：https://github.com/thuml/HashGAN。

## 环境配置

- Python3, NumPy, TensorFlow-gpu, SciPy, Matplotlib, OpenCV, easydict, yacs, tqdm
- NVIDIA GPU

我们提供了一个 `environment.yaml` 文件，你可以简单地使用 `conda env create -f environment.yml` 来创建环境。

或者你可以从头创建环境：
```bash
conda create --no-default-packages -n HashGAN python=3.6 && source activate HashGAN
conda install -y numpy scipy matplotlib tensorflow-gpu opencv
pip install easydict yacs tqdm pillow
```

## 数据准备
在 `data_list/` 文件夹中，我们提供了四个示例来展示如何准备图像训练数据。如果你想添加其他数据集作为输入，你需要像cifar10数据集那样准备 `train.txt`、`test.txt`、`database.txt` 和 `database_nolabel.txt` 文件。

如果你已经有 `database.txt` 文件，你可以通过运行 `change.py` 文件来生成 `database_nolabel.txt` 文件。 

这些data list的参数设置可以在 `config/` 文件夹下的对应配置文件中找到。

除了已经划分好的data list，你还需要自行下载对应的图像集。

## 预训练模型

你可以从[这里](https://github.com/thulab/DeepHash/releases/download/v0.1/reference_pretrain.npy.zip)下载imagenet预训练的Alexnet模型，然后放在 `pretrained_models/` 文件夹下。
`pretrained_models/cifar10/` 文件夹为论文作者提供的预训练的生成器模型，你可以在 `config/cifar_step_2.yaml` 配置文件中看到它被使用。

## 训练

训练过程可以分为两个步骤：
1. 训练图像生成器。
2. 使用原始标注图像和生成的图像微调Alexnet。

在 `config` 文件夹中，以下几个配置文件是论文作者提供的示例。除此之外，我们还创建了用于训练不同位数和迭代步数的模型的配置文件。

```
config
├── cifar_evaluation.yaml
├── cifar_step_1.yaml
├── cifar_step_2.yaml
└── nuswide_step_1.yaml
```

你可以使用如下命令运行模型：

- `python main.py --cfg config/cifar_step_1.yaml --gpus 0`
- `python main.py --cfg config/cifar_step_2.yaml --gpus 0`

你可以使用tensorboard来监控训练过程，如损失和平均精确率（MAP）。训练生成的模型将存在 `output` 文件夹中，在对应模型的目录下，运行命令 `tensorboard --logdir=logs`，即可在浏览器中查看训练过程。


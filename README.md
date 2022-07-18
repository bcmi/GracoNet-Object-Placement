**GracoNet**: Learning Object Placement via Dual-path Graph Completion
========

Official PyTorch Implementation for **GracoNet** (**GRA**ph **CO**mpletion **NET**work).

We treat object placement as a graph completion problem and propose a novel graph completion module (GCM). The background scene is represented by a graph with multiple nodes at different spatial locations with various receptive fields. The foreground object is encoded as a special node that should be inserted at a reasonable place in this graph. We also design a dual-path framework upon GCM to fully exploit annotated composite images, which successfully generates plausible and diversified object placement. GracoNet achieves **0.847** SimOPA accuracy on OPA dataset.

![GracoNet](.github/GracoNet.png)


# Model Zoo
We provide models for TERSE \[[arxiv](https://arxiv.org/abs/1904.05475)\], PlaceNet \[[arXiv](https://arxiv.org/abs/1812.02350)\], and our GracoNet:

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>acc.</th>
      <th>FID</th>
      <th>LPIPS</th>
      <th>url of model & logs</th>
      <th>file name</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TERSE</td>
      <td>0.718</td>
      <td>57.39</td>
      <td>0</td>
      <td>COMING SOON</td>
      <td>terse.zip</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>PlaceNet</td>
      <td>0.724</td>
      <td>41.26</td>
      <td>0.198</td>
      <td>COMING SOON</td>
      <td>placenet.zip</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>GracoNet</td>
      <td>0.847</td>
      <td>27.75</td>
      <td>0.206</td>
      <td><a href="https://cloud.bcmi.sjtu.edu.cn/sharing/5y74jrw2a">bcmi cloud</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1qzEAjHjSarvst5eY3V2Xaw">baidu disk</a>&nbsp;(code: 8rqm)</td>
      <td>graconet.zip</td>
      <td>184Mb</td>
    </tr>
  </tbody>
</table>

We plan to include more models in the future.


# Usage
We provide instructions on how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/bcmi/GracoNet-Object-Placement.git
```
Then, create a virtual environment:
```
conda create -n graconet python=3.6
conda activate graconet
```
Install PyTorch 1.9.1 (require CUDA >= 10.2):
```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
```
Install other packages:
```
pip install -r requirements.txt
```
Build faster-rcnn for SimOPA metric evaluation:
```
cd faster-rcnn/lib
python setup.py build develop
cd ../..
```

## Data preparation
Download and extract OPA dataset from [bcmi cloud](https://cloud.bcmi.sjtu.edu.cn/sharing/anOViiqDN) or [baidu disk](https://pan.baidu.com/s/1tl0x55osXG5hNdIaW_ysuQ) (code: a2ux). We expect the directory structure to be the following:
```
<PATH_TO_OPA>
  bg/                # background images
  fg/                # foreground images with masks
  com_pic/           # composite images with masks
  train_data.csv     # train annotation
  test_data.csv      # test annotation
```

**Note**: The above directory structure is different from the officially released OPA dataset \[ [Github](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA) | [arXiv](https://arxiv.org/pdf/2107.01889.pdf) \]. However, the two versions are essentially the same.

Then, make some preprocessing:
```
python tool/preprocess.py --data_root <PATH_TO_OPA>
```
You will see some new files and directories:
```
<PATH_TO_OPA>
  com_pic_testpos299/          # test set positive composite images (resized to 299)
  test_data_pos.csv            # test annotation for positive samples
  test_data_pos_unique.csv     # test annotation for positive samples with different fg/bg pairs 
```

## Training
To train GracoNet on a single 24GB gpu with batch size 32 for 11 epochs, run:
```
python main.py --data_root <PATH_TO_OPA> --expid <YOUR_EXPERIMENT_NAME>
```
To see the change of losses dynamically, use TensorBoard:
```
tensorboard --logdir result/<YOUR_EXPERIMENT_NAME>/tblog --port <YOUR_SPECIFIED_PORT>
```

## Inference
To predict composite images from a trained GracoNet model, run:
```
python infer.py --expid <YOUR_EXPERIMENT_NAME> --epoch <EPOCH_TO_EVALUATE> --eval_type eval
python infer.py --expid <YOUR_EXPERIMENT_NAME> --epoch <EPOCH_TO_EVALUATE> --eval_type evaluni --repeat 10
```
For example, if you want to infer our best GracoNet model, please 1) download ```graconet.zip``` given in the model zoo, 2) place it under ```result``` and uncompress it:
```
mv path/to/your/downloaded/graconet.zip result/graconet.zip
cd result
unzip graconet.zip
cd ..
```
and 3) run:
```
python infer.py --expid graconet --epoch 11 --eval_type eval
python infer.py --expid graconet --epoch 11 --eval_type evaluni --repeat 10
```

## Evaluation
To evaluate SimOPA accuracy, please 1) download the faster-rcnn model pretrained on visual genome from this [link](https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view) (provided by [Faster-RCNN-VG](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome)) to ```faster-rcnn/models/faster_rcnn_res101_vg.pth```, 2) download the SimOPA model from [bcmi cloud](https://cloud.bcmi.sjtu.edu.cn/sharing/XPEgkSHdQ) or [baidu disk](https://pan.baidu.com/s/1skFRfLyczzXUpp-6tMHArA) (code: 0qty) (provided by [OPA](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA)) to ```SIMOPA_MODEL_PATH```, and 3) run:
```
sh script/eval_acc.sh <YOUR_EXPERIMENT_NAME> <EPOCH_TO_EVALUATE> <SIMOPA_MODEL_PATH>
```
To evaluate FID score, run:
```
sh script/eval_fid.sh <YOUR_EXPERIMENT_NAME> <EPOCH_TO_EVALUATE> <PATH_TO_OPA/com_pic_testpos299>
```
To evaluate LPIPS score, run:
```
sh script/eval_lpips.sh <YOUR_EXPERIMENT_NAME> <EPOCH_TO_EVALUATE>
```
To collect evaluation results of different metrics, run:
```
python tool/sum_eval.py --expid <YOUR_EXPERIMENT_NAME> --eval_type eval
python tool/sum_eval.py --expid <YOUR_EXPERIMENT_NAME> --eval_type evaluni
```
You will find results summarized under ```result/YOUR_EXPERIMENT_NAME/```.


# Acknowledgement
Some of the evaluation codes in this repo are borrowed and modified from [Faster-RCNN-VG](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome), [OPA](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA), [FID-Pytorch](https://github.com/mseitzer/pytorch-fid), and [Perceptual Similarity](https://github.com/richzhang/PerceptualSimilarity). Thanks them for their great work.
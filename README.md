# SOLNet

This repository contains the PyTorch implementation of the paper "Speed-oriented Lightweight Salient Object Detection in Optical Remote Sensing Images" published in TGRS. The code provided here is designed to reproduce the experimental results and figures presented in the paper.  
![framework_overview](./fig/framework_overview.png)  

# Requirements

    python==3.7.12
    pytorch==1.11.0
    torchvision==0.12.0
    torchaudio==0.11.0
    cudatoolkit==11.3
    tensorboard==1.15.0
    tqdm==4.66.1
    thop
    imageio
    numpy
    pyyaml

## Dataset Source

The dataset used for training and testing the model in this repository is sourced from the following location. 

| Dataset Name | Source URL |
|-------------|-----------|
| ORSSD   | [Link to ORSSD](https://pan.baidu.com/s/1k44UlTLCW17AS0VhPyP7JA) |
| EORSSD   | [Link to EORSSD](https://github.com/rmcong/EORSSD-dataset) |
| ors-4199 (Code: fy06)   | [Link to ors-4199](https://pan.baidu.com/share/init?surl=ZWVSzFpRjN4BK-c9hL6knQ) |

## Customizing Training Configuration

You have the flexibility to tailor the training process to your specific needs by modifying the configuration parameters in the `SOLNet.yaml` file located within the `config` directory. This allows you to adjust settings such as learning rates, batch sizes, and other hyperparameters to optimize training for your dataset and hardware setup.

## Running the Training Script

Make sure your data is organized correctly, as this will impact the training process.

```
python train.py
```

## Analyzing Training Results with TensorBoard

After the training process is completed, you can analyze the model's performance by examining the curves in TensorBoard. This visual representation of the training metrics will help you determine the optimal weights for your model.

1. **Launch TensorBoard**:
   Run the following command to launch TensorBoard and visualize the training metrics:

```
tensorboard --logdir=[path_to_your_logs]
```
 
Replace [path_to_your_logs] with the actual path to the directory containing your TensorBoard logs.

3. **Identify Optimal Weights**:
   Once TensorBoard is open, navigate to the sections displaying the training loss, accuracy, or other relevant metrics. Use these curves to identify the epoch or checkpoint that corresponds to the best performance of your model.

4. **Run the Model Conversion Script**:
Use the provided model conversion script to convert the identified optimal weights. Execute the following command:  

```
cd ./model
python Model_Convert.py
```

Before proceeding with the model conversion, it is essential to verify that the `SOLNet.yaml` file located in the `config` directory contains the correct paths for both checkpoint storage and model conversion. Incorrect paths may lead to failures in converting the model. Please ensure that the following sections in `SOLNet.yaml` are correctly configured:

- `checkpoint_path`: The directory where model checkpoints are saved.
- `model_convert`: The path where the converted model will be stored.
  
## Running the Testing Script

Once the model conversion process is complete, you could verify the functionality and performance of the converted model. To do this, you should run the following test code:

```
python test.py
```

## Results

Our prediction results on EORSSD and ORSSD datasets are available for download from [Google Cloud Drive](https://drive.google.com/file/d/1L5-YBXdrrurq2TN495ecy4JuK7Hxoq-p/view?usp=sharing).

## Evaluation Tool

After obtaining the predictions from your model, you can leverage [MATLAB evaluation tools](https://github.com/MathLee/MatlabEvaluationTools) to assess the performance of your model's predictions. MATLAB offers a comprehensive set of functions and metrics that can help you quantify the accuracy, precision, recall, and other relevant performance indicators of your model.

# Citation

```
@ARTICLE{10772137,
  author={Li, Zhaoyang and Miao, Yinxiao and Li, Xiongwei and Li, Wenrui and Cao, Jie and Hao, Qun and Li, Dongxing and Sheng, Yunlong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Speed-oriented Lightweight Salient Object Detection in Optical Remote Sensing Images}, 
  year={2024},
  doi={10.1109/TGRS.2024.3509725}}
```

# Acknowledgment
This project has to some extent been informed by the efforts of several open-source projects, with [RepVGG](https://github.com/DingXiaoH/RepVGG) and [ODConv](https://github.com/OSVAI/ODConv/tree/main) being part of the references. We are grateful for their contributions to the open-source community and encourage users of our project to also consider citing these projects when utilizing our code. Please be aware that while this project is governed by the [MIT](LICENSE), the use of the referenced projects' code must comply with their respective licenses. Users are advised to review these licenses to ensure proper compliance and understand that they are solely responsible for any legal implications arising from the use of the code. We appreciate your respect for the intellectual property rights of all contributors and recommend seeking legal counsel if you have any questions regarding licensing.

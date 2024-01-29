# Distillation for High-Quality Knowledge Extraction via Explainable Oracle Approach


PyTorch official implementation of (Myunghak Lee, Wooseong Cho, Sungsik Kim, Jinkyu Kim, and Jaekoo Lee. "Distillation for High-Quality Knowledge
Extraction via Explainable Oracle Approach" BMVC, 2023).

## Description
![image](https://github.com/myunghakLee/Distillation-for-High-Quality-Knowledge-Extraction/assets/12128784/731f6f9f-cb9f-4e6b-b358-53b9ecb972c3)

An overview of our proposed knowledge distillation method, which consists of two main steps: (A) Generating Relevance-Reinforced Inputs and (B) Transfer Knowledge via Oracle Teacher Model. In Step (A), we generate $\mathbf{x}^*$ where input pixels that make the model correctly classify are reinforced. And in Step (B) this reinforced data is then used to extract the teacher model's responses for the classification task, transferring them into the student model.

Although our model can show very high accuracy, it cannot be used in real situations because ground truth information must be known in advance. So we can't use oracle model directly in real situation. Therefore, we will use the knowledge selected by the oracle teacher model to learn a student model that can make inferences without prior information about the ground truth.

![image](https://github.com/myunghakLee/Distillation-for-High-Quality-Knowledge-Extraction/assets/12128784/3232ccb1-88a6-41a5-9430-bb6fb976e6d5)
The Oracle teacher model not only has high performance, but also has high quality response knowledge. As illustrated in the figure above, t-sne with the response knowledge of the oracle teacher model($\gamma > 0$) shows better clustering than the response knowledge of the scratch model($\gamma = 0$). 

![image](https://github.com/myunghakLee/Distillation-for-High-Quality-Knowledge-Extraction/assets/12128784/d2c0fa41-c528-40a4-b914-5441554fcc3e)
Nevertheless, the amount of information(i.e., ECE) extracted from the oracle teacher model is small(illustrated in the figure above). In other words, necessary information (information about the target class and information about similarities between classes) is preserved, and the total amount of information is low, so it is advantageous for knowledge transfer.

## Requirements
- PyTorch (> 1.2.0)
- torchvision
- numpy

## How to Use
We provide a trained teacher model. Therefore, you only need to train the student model using the code below.
```
python main.py \
    --data_dir ./data --data CIFAR100 \
    --save_dir_name DeleteMe \
    --model_t resnet20 --model_s resnet20 \
    --temperature 1.5 --lrp_gamma 0.2 \
    --ce_weight 1.0 --alpha 1.0 \
    --weight_decay 0.0001
```

If you want to test the student model, use it.
```
python test.py --data_dir data --data CIFAR100 \
               --model_path student_models/CIFAR100/Hetero/resnet20_2_resnet20/best.pth \
               --model resnet20
```

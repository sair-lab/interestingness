# Visual Interestingness

* Refer to the [project description](https://theairlab.org/interestingness) for more details.

* This code is for the following [paper](https://arxiv.org/pdf/2005.08829.pdf), which is selected for **oral presentation** (2%) at ECCV 2020.

     [Chen Wang](https://chenwang.site), [Wenshan Wang](http://www.wangwenshan.com/), [Yuheng Qiu](https://theairlab.org/team/yuheng/), [Yafei Hu](https://theairlab.org/team/yafeih/), and [Sebatian Scherer](https://www.ri.cmu.edu/ri-faculty/sebastian-scherer), [Visual Memorability for Robotic Interestingness via Unsupervised Online Learning](https://arxiv.org/pdf/2005.08829.pdf), European Conference on Computer Vision (ECCV), 2020.

* We also provide ROS wrapper for this project, you may go to [interestingness_ros](https://github.com/wang-chen/interestingness_ros).

---
## Install Dependencies

      pip3 install -r requirements.txt

---
## Long-term Learning

* You may skip this step, if you download the pre-trained [vgg16.pt](https://github.com/wang-chen/interestingness_v2/releases/download/v1.0/vgg16.pt) into folder "saves".


* Download [coco](http://cocodataset.org) dataset into folder [data-root]:

      bash download_coco.sh [data-root] # replace [data-root] by your desired location

     The dataset will be look like:

      data-root
      ├──coco
         ├── annotations
         │   ├── annotations_trainval2017
         │   └── image_info_test2017
         └── images
             ├── test2017
             ├── train2017
             └── val2017

* Run

      python3 longterm.py --data-root [data-root] --model-save saves/vgg16.pt
      
      # This requires a long time for training on single GPU.
      # Create a folder "saves" manually and a model named "ae.pt" will be saved.


## Short-term Learning

* Dowload the [SubT](http://theairlab.org/dataset/interestingness) front camera data (SubTF) and put into folder "data-root", so that it looks like:

      data-root
      ├──SubTF
         ├── 0817-ugv0-tunnel0
         ├── 0817-ugv1-tunnel0
         ├── 0818-ugv0-tunnel1
         ├── 0818-ugv1-tunnel1
         ├── 0820-ugv0-tunnel1
         ├── 0821-ugv0-tunnel0
         ├── 0821-ugv1-tunnel0
         ├── ground-truth
         └── train

* Run

      python3 shortterm.py --data-root [data-root] --model-save saves/vgg16.pt --dataset SubTF --memory-size 1000 --save-flag n1000
      
      # This will read the previous model "ae.pt".
      # A new model "ae.pt.SubTF.n1000.mse" will be generated.
 
* You may skip this step, if you download the pre-trained [vgg16.pt.SubTF.n1000.mse](https://github.com/wang-chen/interestingness_v2/releases/download/v1.0/vgg16.pt.SubTF.n1000.mse) into folder "saves".
 
 
## On-line Learning
 
 * Run
 
         python3 online.py --data-root [data-root] --model-save saves/vgg16.pt.SubTF.n1000.mse --dataset SubTF --test-data 0 --save-flag vgg16

         # --test-data The sequence ID in the dataset SubTF, [0-6] is avaiable
         # This will read the trained model "vgg16.pt.SubTF.n1000.mse" from short-term learning.
         
 * Alternatively, you may test all sequences by running
 
         bash test.sh
 
 * This will generate results files in folder "results".

 * You may skip this step, if you download our generated [results](https://github.com/wang-chen/interestingness/releases/download/v1.0/results.zip).

---
## Evaluation

* We follow the [SubT](https://github.com/wang-chen/SubT.git) tutorial for evaluation, simply run

      python performance.py --data-root [data-root] --save-flag vgg16 --category normal --delta 1 2 3
      # mean accuracy: [0.66410684 0.83252872 0.91957693]
 
      python performance.py --data-root [data-root] --save-flag vgg16 --category difficult --delta 1 2 4
      # mean accuracy: [0.4325405  0.59426775 0.7577122 ]
      
* This will generate performance figures and create data curves for two categories in folder "performance".

---
## Citation

      @inproceedings{wang2020visual,
        title={{Visual Memorability for Robotic Interestingness via Unsupervised Online Learning}},
        author={Wang, Chen and Wang, Wenshan and Qiu, Yuheng and Hu, Yafei and Scherer, Sebastian},
        booktitle={European Conference on Computer Vision (ECCV 2020)},
        year={2020},
      }

* Download [this paper](https://arxiv.org/pdf/2005.08829.pdf).

---
You may watch the following video to catch the idea of this work.

[<img src="https://img.youtube.com/vi/PXIcm17fEko/maxresdefault.jpg" width="100%">](https://youtu.be/PXIcm17fEko)



# Dependencies
   Matplotlib, PyTorch, TorchVision, OpenCV
   
      conda install -c conda-forge matplotlib
      conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
      conda install -c conda-forge opencv

## Long-term Learning

* Download [coco](http://cocodataset.org) dataset into folder "data-root", so that it looks like:

      data-root
      ├──coco
         ├── annotations
         │   ├── annotations_trainval2017
         │   └── image_info_test2017
         └── images
             ├── test2017
             ├── train2017
             └── val2017


* Install coco dataset tools (required by PyTorch).

      conda install -c conda-forge pycocotools

* Run

      python3 train_coder.py --data-root [data-root] --model-save saves/ae.pt
      
      # This requires a long time for training on single GPU.
      # Create a folder "saves" manually and a model named "ae.pt" will be saved.

* You may skip this step, if you download the pre-trained [at.pt](link).


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

      python3 train_interest.py --data-root [data-root] --model-save saves/ae.pt --dataset SubTF --memory-size 1000 --save-flag n1000
      
      # This will read the previous model "ae.pt".
      # A new model "ae.pt.SubTF.n1000.mse" will be generated.
 
* You may skip this step, if you download the pre-trained [ae.pt.SubTF.n1000.mse](link).
 
 
 ## On-line Learning
 
 * Run
 
         python3 test_interest.py --data-root [data-root] --model-save saves/ae.pt.SubTF.n1000.mse --dataset SubTF --test-data 0

         # --test-data The sequence ID in the dataset SubTF, [0-6] is avaiable
         # This will read the trained model "ae.pt.SubTF.n1000.mse" from short-term learning.
         
 * For convenience, you may run
 
         bash test.sh
 
 * This will generate results files in folder "results", that is compatible with evaluation tools in [SubT](https://github.com/wang-chen/SubT.git)

 * You may skip this step, if you download our generated [results](link).


# Evaluation

* We follow the [SubT](https://github.com/wang-chen/SubT.git) tutorial for evaluation, simply run

      python performance.py --data-root [data-root] --save-flag n1000 --category interest-1
      # mean accuracy: [0.66235087 0.84281507 0.95655934]
 
      python performance.py --data-root [data-root] --save-flag n1000 --category interest-2
      # mean accuracy: [0.40703316 0.58456123 0.76820896]
      
* This will generate performance figures and create data curves for two categories in folder "performance".


# Citation

      @article{wang2020visual,
        author = {Wang, Chen and Wang, Wenshan and Qiu, Yuheng and Hu, Yafei and Scherer, Sebastian},
        journal = {arXiv preprint arXiv:2005.08829},
        title = {{Visual Memorability for Robotic Interestingness via Unsupervised Online Learning}},
        year = {2020}
      }

---
You may watch the following video to catch the idea of this work.

[<img src="https://img.youtube.com/vi/gBBdYdUrIcw/maxresdefault.jpg" width="100%">](https://youtu.be/gBBdYdUrIcw)

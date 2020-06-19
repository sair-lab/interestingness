
You may watch the following video to catch the idea of this work.

   [<img src="https://img.youtube.com/vi/gBBdYdUrIcw/maxresdefault.jpg" width="100%">](https://youtu.be/gBBdYdUrIcw)

Dependencies:

      conda install -c conda-forge matplotlib
      conda install -c conda-forge opencv
      conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

## Long-term Learning

* You may skip long-term learning, if you download the pre-trained model.

* Obtain coco dataset and install its tools

      conda install -c conda-forge pycocotools

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

* Training:

      python3 train_coder.py --data-root [data-root] --model-save saves/ae.pt
      
      # This requires a long time for training on single GPU.
      # Create a folder "saves" manually and a model named "ae.pt" will be saved.


## Short-term Learning

* Dowload the [SubT](http://theairlab.org/dataset/interestingness) front camear images (SubTF) and put into folder "data-root", so that it looks like:

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

* Train 

      python train_interest.py --data-root [data-root] --model-save saves/ae.pt --dataset SubTF
      
      # This will read the previous model "ae.pt".
      # A new model "ae.pt.SubTF.interest.mse" will be generated.
 


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

      python3 train_coder.py --data-root [data-root] --save saves/at.pt

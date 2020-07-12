cd $1
mkdir coco && cd coco
mkdir images
mkdir annotations

cd annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip annotations_trainval2017.zip && mv annotations annotations_trainval2017
unzip image_info_test2017.zip  && mv annotations image_info_test2017

cd .. && cd images
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip train2017.zip

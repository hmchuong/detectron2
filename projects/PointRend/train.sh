export DETECTRON2_DATASETS=/home/ubuntu/detectron_dataset
python train_net.py --config-file configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml --num-gpus 1
#!/usr/bin/env bash
subset=$1
subset=${subset:-0}
echo "subset: $subset"
python main/predict.py -j 4 --gpu-id 0 --arch ResNetDCT_Upscaled_Static --subset $subset --resume pretrained/resnet50dct_upscaled_static_$subset/minloss_model.pth --data ./data 
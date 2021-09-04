#! /bin/bash
# echo -n "Enter Main Model Path: "
# read MAIN_MODEL_PATH
# echo -n "Enter Sub Model Path: "
# read SUB_MODEL_PATH

# ViT
python3 train.py -c config_main.json
# AgeModel
python3 train.py -c config_sub.json

# Submit
# python3 submit.py -c config_agemodel.json -m $MAIN_MODEL_PATH -s $SUB_MODEL_PATH

python3 submit.py -c config_submit_main.json

python3 submit.py -c config_submit_sub.json

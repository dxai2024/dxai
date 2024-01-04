"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

FILE=$1

if  [ $FILE == "celeba-hq-dataset" ]; then
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=../Data/celeba_hq.zip
    mkdir -p ../Data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ../Data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-dataset" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=../Data/afhq.zip
    mkdir -p ../Data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ../Data
    rm $ZIP_FILE

elif  [ $FILE == "pretrained-dxai-afhq" ]; then
    mkdir -p ./expr/checkpoints_afhq_pretrained
    CKPT_FILE='./expr/checkpoints_afhq_pretrained/320001_nets_ema.ckpt'
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1F2H0Fpq60pFksSU6FAwnCnJbKMZ6hLAF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F2H0Fpq60pFksSU6FAwnCnJbKMZ6hLAF" -O $CKPT_FILE && rm -rf /tmp/cookies.txt
  
elif  [ $FILE == "pretrained-dxai-celeba-hq" ]; then
    mkdir -p ./expr/checkpoints_celeba_hq_pretrained
    CKPT_FILE='./expr/checkpoints_celeba_hq_pretrained/320001_nets_ema.ckpt'
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1N972J9AF1X8-uuIcfqKtWpGn0OSAM2Db' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1N972J9AF1X8-uuIcfqKtWpGn0OSAM2Db" -O $CKPT_FILE && rm -rf /tmp/cookies.txt

    
elif  [ $FILE == "pretrained-resnet18-afhq" ]; then
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12MVga-DdXbhLk7RoKT9wfjKTnKSs7Hw5' -O afhq_resnet18_ch_3_weights.ckpt

elif  [ $FILE == "pretrained-resnet18-celeba-hq" ]; then
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S7yPXhkk-eK7YCvcNVjFNC6xuBPSkM2D' -O celeba_hq_resnet18_ch_3_weights.ckpt
        
else
    echo "Available arguments are pretrained-network-celeba-hq, pretrained-network-afhq, celeba-hq-dataset, and afhq-dataset."
    exit 1

fi

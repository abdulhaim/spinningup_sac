#!/bin/bash

FOLDER_NAME="spinningup"
declare -a arr=(
    "ec2-18-208-216-83.compute-1.amazonaws.com"
    )

for SSH_ADDRESS in "${arr[@]}"
do
    echo $SSH_ADDRESS

    # Pass folder that I want to train
    ssh -i ~/Documents/abdulhai.pem ubuntu@$SSH_ADDRESS "mkdir /home/ubuntu/$FOLDER_NAME"
    scp -i ~/Documents/abdulhai.pem -r ~/Desktop/sac-baseline-experiments/spinningup ubuntu@$SSH_ADDRESS:/home/ubuntu/

done

scp -i ~/Documents/abdulhai.pem -r ubuntu@ec2-18-208-216-83.compute-1.amazonaws.com:/home/ubuntu/spinningup/data .

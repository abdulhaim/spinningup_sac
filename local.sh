#!/bin/bash

FOLDER_NAME="spinningup"
declare -a arr=(
    "ec2-54-82-226-198.compute-1.amazonaws.com"
    )

for SSH_ADDRESS in "${arr[@]}"
do
    echo $SSH_ADDRESS

    # Pass folder that I want to train
    ssh -i ~/Documents/abdulhai.pem ubuntu@$SSH_ADDRESS "mkdir /home/ubuntu/$FOLDER_NAME"
    scp -i ~/Documents/abdulhai.pem -r ~/Desktop/sac-baseline-experiments/spinningup ubuntu@$SSH_ADDRESS:/home/ubuntu/

done

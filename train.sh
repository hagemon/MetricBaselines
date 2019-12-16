#!/bin/bash

# for CUB
#python -u train.py --method Triplet
#python -u train.py --method N_pair --balanced --instances 2 --batch-size 120 --cm
#python -u train.py --method Lifted --batch-size 120
#python -u train.py --method Angular --balanced --instances 2 --batch-size 120
python -u train.py --method RankedList --balanced --instances 3 --batch-size 120 --iteration 1000

# for CARS196
# python -u train.py --method Triplet --lr 1e-4 --dataset CARS196
# python -u train.py --method N_pair --balanced --instances 2 --batch-size 120 --cm --lr 1e-4 --dataset CARS196
# python -u train.py --method Lifted --batch-size 120 --lr 1e-4 --dataset CARS196
# python -u train.py --method Angular --balanced --instances 2 --batch-size 120 --lr 1e-4 --dataset CARS196
python -u train.py --method RankedList --balanced --instances 3 --batch-size 120 --lr 1e-4 --dataset CARS196 --iteration 1000

# for SOP

#python -u train.py --method Triplet --lr 1e-4 --dataset SOP
#python -u train.py --method N_pair --balanced --instances 2 --batch-size 120 --cm --lr 1e-4 --dataset SOP
# python -u train.py --method Lifted --batch-size 120 --lr 1e-4 --dataset SOP
#python -u train.py --method Angular --balanced --instances 2 --batch-size 120 --lr 1e-4 --dataset SOP
# python -u train.py --method RankedList --balanced --instances 3 --batch-size 120 --lr 1e-4 --dataset SOP

#!/bin/bash

## Run GalilAI method and collect PNN training data
python plan_mujoco.py --action_mode walker --experiment gravity --output walker-trainMass-testGravity

## Collect training data for 5 baseline seeds
python plan_mujoco.py --action_mode walker --experiment gravity --baseline_only True --output walker-trainMass-testGravity

## Train PNNs for baseline 
python train_pnn.py --action_mode walker --base_path walker-trainMass-testGravity

## Calculate disagreement between (seen, unseen) causal envs
python train_pnn.py --action_mode walker --base_path walker-trainMass-testGravity --compare_only True

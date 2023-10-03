#!/bin/bash

input_variables=("/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D2/1_forward_normal_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D2/2_backward_normal_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D2/3_left_and_right_normal_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D2/4_diagonal_normal_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D2/5_mixed_normal_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D3/6_forward_small_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D3/7_backward_small_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D3/8_left_and_right_small_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D3/9_diagonal_small_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D3/10_mixed_small_step/data.log"
"/home/iiticublap185/git/adherent_ergocub/datasets/mocap/D3/11_mixed_normal_and_small_step/data.log")

# Loop through the input variables and run Python scripts
for var in "${input_variables[@]}"; do
    echo "Running Python script with input: $var"
    python3 retargeting.py --KFWBGR --filename "$var" --save --deactivate_visualization # --smaller_forward_steps
    python3 retargeting.py --KFWBGR --mirroring --filename "$var" --save --deactivate_visualization # --smaller_forward_steps
done
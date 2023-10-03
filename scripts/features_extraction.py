# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# TODO: manually set the env variable at each restart of the docker container
# export IGN_FILE_PATH=/home/pviceconte/git/adherent_ergocub/src/adherent/model/ergoCubGazeboV1_xsens/:$IGN_FILE_PATH

import os
import json
import argparse
import time

import numpy as np
from scenario import gazebo as scenario
from adherent.data_processing import utils
from gym_ignition.utils.scenario import init_gazebo_sim
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.data_processing import features_extractor

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

# Our custom dataset is divided in two datasets: D2 and D3
parser.add_argument("--dataset", help="Select a dataset between D2 and D3.", type=str, default="D2")

# Each dataset is divided into portions. D2 includes portions [1,5]. D3 includes portions [6,11].
parser.add_argument("--portion", help="Select a portion of the chosen dataset. Available choices: from 1 to 5 for D2,"
                                      "from 6 to 11 for D3.", type=int, default=1)

# Each portion of each dataset has been retargeted as it is or mirrored. Select if you want to visualize the mirrored version
parser.add_argument("--mirrored", help="Visualize the mirrored version of the selected dataset portion.",action="store_true")

# Plot configuration
parser.add_argument("--plot_global", help="Visualize the computed global features.",action="store_true")
parser.add_argument("--plot_local", help="Visualization the computed local features.",action="store_true")

# Store configuration
parser.add_argument("--save", help="Store the network input and output vectors in json format.",action="store_true")

# How many times replicate stop frames
parser.add_argument("--replicate_stop_frames", help="How many times you want to replicate the stop frames.", type=int, default=0)

# Skip some frames
parser.add_argument("--skip_annotated_frames", help="Skip the annotated frames.",action="store_true")

args = parser.parse_args()

dataset = args.dataset
retargeted_mocap_index = args.portion
mirrored = args.mirrored
plot_global = args.plot_global
plot_local = args.plot_local
store_as_json = args.save
replicate_stop_frames = args.replicate_stop_frames
skip_annotated_frames = args.skip_annotated_frames

# ====================
# LOAD RETARGETED DATA
# ====================

# Define the selected subsection of the dataset to be loaded and the correspondent interesting frame interval
if dataset == "D2":
    retargeted_mocaps = {1:"1_forward_normal_step",2:"2_backward_normal_step",3:"3_left_and_right_normal_step",
                         4:"4_diagonal_normal_step",5:"5_mixed_normal_step"}
    limits = {1: [3750, 35750], 2: [1850, 34500], 3: [2400, 36850], 4: [1550, 16000], 5: [2550, 82250]}
    stops = {1: [], 2: [], 3: [], 4: [], 5: []} # TODO: not defined for D2
    skips = {1: [], 2: [], 3: [], 4: [], 5: []}  # TODO: not defined for D2
elif dataset == "D3":
    retargeted_mocaps = {6:"6_forward_small_step",7:"7_backward_small_step",8:"8_left_and_right_small_step",
                         9:"9_diagonal_small_step",10:"10_mixed_small_step",11:"11_mixed_normal_and_small_step"}

    limits = {6: [1500, 28500], 7: [1750, 34000], 8: [2900, 36450], 9: [1250, 17050], 10: [1450, 78420], 11: [1600, 61350]}

    stops =  {6: [[3740,3900], [6080,6250], [10000,10150], [13040,13240], [16000,16130]],
              7: [[5280,5420], [8460,8620], [10980,11130], [13660,13820], [16720,16890], [19690,19810], [20240,20400], [20920,21090], [24830,25070], [30430,30560], [31010,31170]],
              8: [[310,530], [1210,1270], [3460,3670], [5700,5770], [7700,7880], [9640,9780], [10960,11050], [14100,14260], [16250,16320], [18000,18120], [20480,20640], [22040,22270], [23880,24070], [26780,26950], [27980,28110], [30080,30200], [33420,33550]],
              9: [[4670,4860], [5390,5510], [6170,6290], [6940,7050], [9190,9320], [12230,12380], [13300,13380], [14000,14130], [15750,15800]],
              10: [[1310,1420], [4410,4610], [8760,8850], [10710,10910], [13050,13160], [13950,14040], [16660,16810], [20800,21010], [23750,23860], [27850,27940], [29800,30070], [32020,32210], [33220,33440], [35270,35370], [37500,37670], [40680,40780], [41620,41760]],
              11: []}

    skips = {6: [], 7: [], 8: [], 9: [], 10: [], # TODO: : not defined for some portions of D3
             11: [[1120,1570],[2050,2480],[8500,8950],[9450,9570],[9800,10215],[10830,11700],[17040,17440],[20460,20950],[21280,21500],[235540,24300],[27960,30200],[32970,33110],[33690,35040],[37260,37470],[41740,42310],[42940,43360],[45340,45570],[45970,46340],[48810,49830],[51060,51400],[51570,51800],[53730,54660],[57500,58540]], # TODO: defined for D3.11 to skip long sidesteps
             }

initial_frame = limits[retargeted_mocap_index][0]
final_frame = limits[retargeted_mocap_index][1]

# Retrieve the stop frames to be replicated in the features (no need to subtract the initial_frame because of how they have been annotated)
stop_frames = stops[retargeted_mocap_index]
skip_frames = skips[retargeted_mocap_index]

# Define the retargeted mocap path
if not mirrored:
    retargeted_mocap_path = "../datasets/retargeted_mocap/" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_RETARGETED.txt"
else:
    retargeted_mocap_path = "../datasets/retargeted_mocap/" + dataset + "_mirrored/" + retargeted_mocaps[retargeted_mocap_index]  + "_RETARGETED_MIRRORED.txt"
script_directory = os.path.dirname(os.path.abspath(__file__))
retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)

# Load the retargeted mocap data
timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path,
                                                                 initial_frame=initial_frame,
                                                                 final_frame=final_frame)

# ===============
# MODEL INSERTION
# ===============

# Set scenario verbosity
scenario.set_verbosity(scenario.Verbosity_warning)

# Get the default simulator and the default empty world
gazebo, world = init_gazebo_sim()

# Retrieve the robot urdf model
icub_urdf = os.path.join(script_directory, "../src/adherent/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

# Insert the robot in the empty world
icub = utils.iCub(world=world, urdf=icub_urdf)

# Get the robot joints
icub_joints = icub.joint_names()

# Define the joints of interest for the features computation and their associated indexes in the robot joints  list
controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm
controlled_joints_indexes = [icub_joints.index(elem) for elem in controlled_joints]
# controlled_joints_retrieved = [icub_joints[index] for index in controlled_joints_indexes] # This is how to use the indexes

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# Create a KinDynComputations object
kindyn = kindyncomputations.KinDynComputations(model_file=icub_urdf, considered_joints=icub_joints)
kindyn.set_robot_state_from_model(model=icub, world_gravity=np.array(world.gravity()))

# ===================
# FEATURES EXTRACTION
# ===================

# Define robot-specific frontal base and chest directions
frontal_base_dir = utils.define_frontal_base_direction(robot="ergoCubV1")
frontal_chest_dir = utils.define_frontal_chest_direction(robot="ergoCubV1")

# Instantiate the features extractor
extractor = features_extractor.FeaturesExtractor.build(ik_solutions=ik_solutions,
                                                       kindyn=kindyn,
                                                       controlled_joints_indexes=controlled_joints_indexes,
                                                       frontal_base_dir=frontal_base_dir,
                                                       frontal_chest_dir=frontal_chest_dir)
# Extract the features
extractor.compute_features()

# ===========================================
# NETWORK INPUT AND OUTPUT VECTORS GENERATION
# ===========================================

# Generate the network input vector X
if replicate_stop_frames and skip_frames:
    X = extractor.compute_X(stop_frames=stop_frames, replicate_stop_frames=replicate_stop_frames,
                            skip_frames=skip_frames)
elif replicate_stop_frames:
    X = extractor.compute_X(stop_frames=stop_frames, replicate_stop_frames=replicate_stop_frames)
elif skip_annotated_frames:
    X = extractor.compute_X(skip_frames=skip_frames)
else:
    X = extractor.compute_X()

if store_as_json:

    # Define the path to store the input X associated to the selected subsection of the dataset
    if not mirrored:
        input_path = "../datasets/IO_features/inputs_subsampled_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_X.txt"
    else:
        input_path = "../datasets/IO_features/inputs_subsampled_mirrored_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_X_MIRRORED.txt"
    input_path = os.path.join(script_directory, input_path)

    # Store the retrieved input X in a JSON file
    with open(input_path, 'w') as outfile:
        json.dump(X, outfile)

    # Debug
    print("Input features have been saved in", input_path)

# Generate the network output vector Y
if replicate_stop_frames and skip_frames:
    Y = extractor.compute_Y(stop_frames=stop_frames, replicate_stop_frames=replicate_stop_frames,
                            skip_frames=skip_frames)
elif replicate_stop_frames:
    Y = extractor.compute_Y(stop_frames=stop_frames, replicate_stop_frames=replicate_stop_frames)
elif skip_annotated_frames:
    Y = extractor.compute_Y(skip_frames=skip_frames)
else:
    Y = extractor.compute_Y()

if store_as_json:

    # Define the path to store the output Y associated to the selected subsection of the dataset
    if not mirrored:
        output_path = "../datasets/IO_features/outputs_subsampled_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_Y.txt"
    else:
        output_path = "../datasets/IO_features/outputs_subsampled_mirrored_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_Y_MIRRORED.txt"
    output_path = os.path.join(script_directory, output_path)

    # Store the retrieved output Y in a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(Y, outfile)

    # Debug
    print("Output features have been saved in", output_path)

# Temporary to avoid issued in subsequent calls of the script
time.sleep(1)

# =======================================================
# VISUALIZE THE RETARGETED MOTION AND THE GLOBAL FEATURES
# =======================================================

if plot_global:

    input("Press Enter to start the visualization of the GLOBAL features")
    utils.visualize_global_features(global_window_features=extractor.get_global_window_features(),
                                    ik_solutions=ik_solutions,
                                    icub=icub,
                                    controlled_joints_indexes=controlled_joints_indexes,
                                    gazebo=gazebo)

# =======================================================
# VISUALIZE THE RETARGETED MOTION AND THE LOCAL FEATURES
# =======================================================

if plot_local:

    input("Press Enter to start the visualization of the LOCAL features")
    utils.visualize_local_features(local_window_features=extractor.get_local_window_features(),
                                   ik_solutions=ik_solutions,
                                   icub=icub,
                                   controlled_joints_indexes=controlled_joints_indexes,
                                   gazebo=gazebo)

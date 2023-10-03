# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# Use tf version 2.3.0 as 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import json
import argparse
from scenario import gazebo as scenario
from gym_ignition.utils.scenario import init_gazebo_sim
from adherent.data_processing.utils import iCub
from adherent.data_processing.utils import define_feet_frames_and_links
from adherent.trajectory_generation.utils import visualize_generated_motion

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--storage_path", help="Path where the generated trajectory is stored. Relative path from script folder.",
                    type=str, default="../datasets/inference/")
parser.add_argument("--plot_joystick_inputs", help="Activate plot of the joystick inputs.", action="store_true")
parser.add_argument("--plot_com", help="Activate plot of the CoM position and velocity.", action="store_true")

args = parser.parse_args()

storage_path = args.storage_path
plot_joystick_inputs = args.plot_joystick_inputs
plot_com = args.plot_com

# ===============================
# LOAD TRAJECTORY GENERATION DATA
# ===============================

# Define robot-specific feet frames
feet_frames, feet_links = define_feet_frames_and_links(robot="ergoCubV1")

# Define the paths for the generated postural, footsteps, joystick inputs and blending coefficients
script_directory = os.path.dirname(os.path.abspath(__file__))
storage_path = os.path.join(script_directory, storage_path)
postural_path = storage_path + "postural.txt"
footsteps_path = storage_path + "footsteps.txt"
joystick_input_path = storage_path + "joystick_input.txt"

# Load generated posturals
with open(postural_path, 'r') as openfile:
    posturals = json.load(openfile)

# Load generated footsteps
with open(footsteps_path, 'r') as openfile:
    footsteps = json.load(openfile)
    l_footsteps = footsteps[feet_frames["left_foot"]]
    r_footsteps = footsteps[feet_frames["right_foot"]]

# Load joystick inputs (motion and facing directions) associated to the generated trajectory
with open(joystick_input_path, 'r') as openfile:
    joystick_input = json.load(openfile)
    raw_data = joystick_input["raw_data"]

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
icub = iCub(world=world, urdf=icub_urdf)

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

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# ==========================
# VISUALIZE GENERATED MOTION
# ==========================

input("Press Enter to start the visualization of the generated trajectory.")
visualize_generated_motion(icub=icub, controlled_joints_indexes=controlled_joints_indexes,
                           gazebo=gazebo, posturals=posturals,
                           l_footsteps=l_footsteps, r_footsteps=r_footsteps,
                           raw_data=raw_data,
                           plot_joystick_inputs=plot_joystick_inputs,
                           plot_com=plot_com)








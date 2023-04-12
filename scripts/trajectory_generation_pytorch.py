# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# TODO: manually set the env variable at each restart of the docker container
# export IGN_FILE_PATH=/home/pviceconte/git/adherent_ergocub/src/adherent/model/ergoCubGazeboV1_xsens/:$IGN_FILE_PATH

import os
import yarp
import time
import argparse

# TODO: remove (but protobuf error)
import tensorflow.compat.v1 as tf

from gym_ignition.utils.scenario import init_gazebo_sim
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.trajectory_generation import trajectory_generator
from adherent.trajectory_generation.utils import *

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--training_path", help="Path where the training-related data are stored. Relative path from script folder.",
                    type = str, default = "../datasets/training_subsampled_mirrored_D2_D3_20230320-190551/")
parser.add_argument("--stream_every_N_steps", help="Data will be streamed every N steps.",
                    type=int, default=3)
parser.add_argument("--stream_every_N_iterations", help="If no steps are performed, data will be streamed every N iterations.",
                    type=int, default=150) # TODO: tune
parser.add_argument("--plot_footsteps", help="Visualize the footsteps.", action="store_true")
parser.add_argument("--time_scaling", help="Time scaling to be applied to the generated trajectory. Keep it integer.",
                    type=int, default=1)

args = parser.parse_args()

training_path = args.training_path
stream_every_N_iterations = args.stream_every_N_iterations
stream_every_N_steps = args.stream_every_N_steps
plot_footsteps = args.plot_footsteps
time_scaling = args.time_scaling

# ==================
# YARP CONFIGURATION
# ==================

# Initialise YARP
yarp.Network.init()

# Open and connect YARP port to retrieve joystick input
p_in = yarp.BufferedPortBottle()
p_in.open("/joystick_in")
yarp.Network.connect("/joystick_out", p_in.getName())

# Initialization of the joystick raw and processed inputs received through YARP ports
raw_data = [] # motion and facing directions
quad_bezier = []
base_velocities = []
facing_dirs = []

# ===============
# MODEL INSERTION
# ===============

# Set scenario verbosity
scenario.set_verbosity(scenario.Verbosity_warning)

# Get the default simulator and the default empty world
gazebo, world = init_gazebo_sim()

# Retrieve the robot urdf model
script_directory = os.path.dirname(os.path.abspath(__file__))
icub_urdf = os.path.join(script_directory, "../src/adherent/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

# Insert the robot in the empty world
initial_base_height = define_initial_base_height(robot="ergoCubV1")
icub = iCub(world=world, urdf=icub_urdf, position=[0, 0, initial_base_height])

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# Get the robot joints
icub_joints = icub.joint_names()

# Define the joints of interest for the features computation and their associated indexes in the robot joints  list
controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'l_wrist_yaw', 'l_wrist_roll', 'l_wrist_pitch', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'r_wrist_yaw', 'r_wrist_roll', 'r_wrist_pitch'] # right arm
controlled_joints_indexes = [icub_joints.index(elem) for elem in controlled_joints]
# controlled_joints_retrieved = [icub_joints[index] for index in controlled_joints_indexes] # This is how to use the indexes

# Create a KinDynComputations object
kindyn = kindyncomputations.KinDynComputations(model_file=icub_urdf, considered_joints=controlled_joints)
kindyn.set_robot_state_from_model(model=icub, world_gravity=np.array(world.gravity()))

# ==================
# PLOT CONFIGURATION
# ==================

plt.ion()

# ====================
# TRAJECTORY GENERATOR
# ====================

# Trajectory control and trajectory generation rates
generation_rate = 1/50
control_rate = 1/100

# Define robot-specific feet vertices positions in the foot frame
local_foot_vertices_pos = define_foot_vertices(robot="ergoCubV1")

# Define robot-specific initial input X for the trajectory generation neural network
initial_nn_X = define_initial_nn_X(robot="ergoCubV1")

# Define robot-specific initial past trajectory features
initial_past_trajectory_base_pos, initial_past_trajectory_facing_dirs, initial_past_trajectory_base_vel = \
    define_initial_past_trajectory(robot="ergoCubV1")

# Define robot-specific initial base yaw angle
initial_base_yaw = define_initial_base_yaw(robot="ergoCubV1")

# Define robot-specific frontal base and chest directions
frontal_base_dir = define_frontal_base_direction(robot="ergoCubV1")
frontal_chest_dir = define_frontal_chest_direction(robot="ergoCubV1")

# Define robot-specific feet frames and links
feet_frames, feet_links = define_feet_frames_and_links(robot="ergoCubV1")

# Define robot-specific initial support foot and vertex
initial_support_foot, initial_support_vertex = define_initial_support_foot_and_vertex(robot="ergoCubV1")

# Define robot-specific initial feet positions
initial_l_foot_position, initial_r_foot_position = define_initial_feet_positions(robot="ergoCubV1")

# Define robot-specific (and possibly training-dependent) base pitch offset
base_pitch_offset = define_base_pitch_offset(robot="ergoCubV1")

# Instantiate the trajectory generator
generator = trajectory_generator.TrajectoryGenerator.build(icub=icub, gazebo=gazebo, kindyn=kindyn,
                                                           controlled_joints_indexes=controlled_joints_indexes,
                                                           training_path=os.path.join(script_directory, training_path),
                                                           local_foot_vertices_pos=local_foot_vertices_pos,
                                                           feet_frames=feet_frames,
                                                           feet_links=feet_links,
                                                           initial_nn_X=initial_nn_X,
                                                           initial_past_trajectory_base_pos=initial_past_trajectory_base_pos,
                                                           initial_past_trajectory_facing_dirs=initial_past_trajectory_facing_dirs,
                                                           initial_past_trajectory_base_vel=initial_past_trajectory_base_vel,
                                                           initial_base_height=initial_base_height,
                                                           initial_base_yaw=initial_base_yaw,
                                                           frontal_base_direction=frontal_base_dir,
                                                           frontal_chest_direction=frontal_chest_dir,
                                                           initial_support_foot=initial_support_foot,
                                                           initial_support_vertex=initial_support_vertex,
                                                           initial_l_foot_position=initial_l_foot_position,
                                                           initial_r_foot_position=initial_r_foot_position,
                                                           time_scaling=time_scaling,
                                                           generation_rate=generation_rate,
                                                           control_rate=control_rate)

# =========
# MAIN LOOP
# =========

while True:

    # Initialize the mergepoint and plot initial footsteps
    if generator.iteration == 1:

        # Dummy mergepoint initialization
        generator.autoregression.update_mergepoint_state()
        generator.kincomputations.update_mergepoint_state()

        # Plot the initial footsteps
        if plot_footsteps:
            for foot in generator.storage.footsteps:
                generator.plotter.plot_new_footstep(support_foot=foot,
                                                    new_footstep=generator.storage.footsteps[foot][-1])

    # Stream data every N steps (or if the maximum number of iterations without steps is reached)
    if generator.get_n_steps() == stream_every_N_steps or generator.get_n_iterations() == stream_every_N_iterations:

        # TODO: also handle missing deactivation times at the beginning ?
        generator.preprocess_data()

        # TODO: Stream data (now this is just printing)
        generator.stream_data()

        # Reset to the last mergepoint
        generator.reset_from_mergepoint()

        # Debug
        input("Press Enter to proceed.")

    # Update the iteration counter
    generator.update_iteration_counter()

    # Retrieve the network output
    current_output, denormalized_current_output = generator.retrieve_network_output_pytorch()

    # Apply the joint positions and the base orientation from the network output
    joint_positions, joint_velocities, new_base_quaternion = \
        generator.apply_joint_positions_and_base_orientation(denormalized_current_output=denormalized_current_output,
                                                             base_pitch_offset=base_pitch_offset)

    # Update the support foot and vertex while detecting new footsteps
    support_foot, update_footsteps_list = generator.update_support_vertex_and_support_foot_and_footsteps()

    # Plot the new footstep
    if update_footsteps_list and plot_footsteps:

        # Plot the last footstep
        generator.plotter.plot_new_footstep(support_foot=support_foot,
                                            new_footstep=generator.storage.footsteps[support_foot][-1])

    # Compute kinematically-feasible base position and updated posturals
    new_base_postural, new_joints_pos_postural, new_joints_vel_postural, \
    new_com_pos_postural, new_com_vel_postural, new_centroidal_momentum_postural = \
        generator.compute_kinematically_fasible_base_and_update_posturals(joint_positions=joint_positions,
                                                                          joint_velocities=joint_velocities,
                                                                          base_quaternion=new_base_quaternion,
                                                                          controlled_joints=controlled_joints)

    # Retrieve user input data from YARP port
    quad_bezier, base_velocities, facing_dirs, raw_data = \
        generator.retrieve_joystick_inputs(input_port=p_in,
                                           quad_bezier=quad_bezier,
                                           base_velocities=base_velocities,
                                           facing_dirs=facing_dirs,
                                           raw_data=raw_data)

    # Use in an autoregressive fashion the network output and blend it with the user input
    blended_base_positions, blended_facing_dirs, blended_base_velocities = \
        generator.autoregression_and_blending(current_output=current_output,
                                              denormalized_current_output=denormalized_current_output,
                                              quad_bezier=quad_bezier,
                                              facing_dirs=facing_dirs,
                                              base_velocities=base_velocities)

    # Update postural storage
    generator.update_storages(base_postural=new_base_postural,
                              joints_pos_postural=new_joints_pos_postural,
                              joint_vel_postural=new_joints_vel_postural,
                              com_pos_postural=new_com_pos_postural,
                              com_vel_postural=new_com_vel_postural,
                              centroidal_momentum_postural=new_centroidal_momentum_postural)

    # Slow down visualization
    time.sleep(0.001)

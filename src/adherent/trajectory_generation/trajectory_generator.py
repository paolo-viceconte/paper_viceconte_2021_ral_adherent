# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn
from mann_pytorch.MANN import MANN

import os
import math
import json
import yarp
import random
import numpy as np
import manifpy as manif
from typing import List, Dict
from scenario import gazebo as scenario
from dataclasses import dataclass, field
from gym_ignition.rbd.idyntree import numpy
from gym_ignition.rbd.conversions import Rotation
from gym_ignition.rbd.conversions import Transform
from adherent.MANN.utils import denormalize
from gym_ignition.rbd.conversions import Quaternion
from adherent.MANN.utils import read_from_file
from adherent.data_processing.utils import iCub
import bipedal_locomotion_framework.bindings as blf
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.data_processing.utils import rotation_2D
from adherent.trajectory_generation.utils import trajectory_blending
from adherent.trajectory_generation.utils import load_output_mean_and_std
from adherent.trajectory_generation.utils import compute_angle_wrt_x_positive_semiaxis
from adherent.trajectory_generation.utils import load_component_wise_input_mean_and_std
from mann_pytorch.utils import get_latest_model_path

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt


@dataclass
class StorageHandler:
    """Class to store all the quantities relevant in the trajectory generation pipeline and save data."""

    # Storage paths for the footsteps, postural, joystick input and blending coefficients
    footsteps_path: str
    postural_path: str
    joystick_input_path: str
    blending_coefficients_path: str

    # Time scaling factor for the trajectory
    time_scaling: int
    generation_to_control_time_scaling: int

    # Auxiliary variables
    feet_frames: Dict
    initial_l_foot_position: List
    initial_r_foot_position: List

    # Storage dictionaries for footsteps, postural, joystick input and blending coefficients
    footsteps: Dict
    posturals: Dict = field(default_factory=lambda: {'base': [], 'joints_pos': [], 'joints_vel': [], 'links': [], 'com_pos': [], 'com_vel': [], 'centroidal_momentum': []})
    joystick_inputs: Dict = field(default_factory=lambda: {'raw_data': [], 'quad_bezier': [], 'base_velocities': [], 'facing_dirs': []})
    blending_coeffs: Dict = field(default_factory=lambda: {'w_1': [], 'w_2': [], 'w_3': [], 'w_4': []})

    # Store footsteps and posturals at the mergepoint
    mergepoint_footsteps: Dict = field(default_factory=lambda: {})
    # mergepoint_posturals: Dict = field(default_factory=lambda: {}) # TODO

    @staticmethod
    def build(storage_path: str,
              feet_frames: Dict,
              initial_l_foot_position: List,
              initial_r_foot_position: List,
              generation_to_control_time_scaling: int,
              time_scaling: int = 1) -> "StorageHandler":
        """Build an instance of StorageHandler."""

        # Storage paths for the footsteps, postural, joystick input and blending coefficients
        footsteps_path = os.path.join(storage_path, "footsteps.txt")
        postural_path = os.path.join(storage_path, "postural.txt")
        joystick_input_path = os.path.join(storage_path, "joystick_input.txt")
        blending_coefficients_path = os.path.join(storage_path, "blending_coefficients.txt")

        # Define initial left footstep
        initial_left_footstep = {}
        initial_left_footstep["pos"] = initial_l_foot_position
        initial_left_footstep["quat"] = [1,0,0,0]
        initial_left_footstep["activation_time"] = 0.0
        initial_left_footstep["deactivation_time"] = -1

        # Define initial right footstep
        initial_right_footstep = {}
        initial_right_footstep["pos"] = initial_r_foot_position
        initial_right_footstep["quat"] = [1,0,0,0]
        initial_right_footstep["activation_time"] = 0.0
        initial_right_footstep["deactivation_time"] = -1

        # Initialize footsteps with initial left and right footsteps
        footsteps = {feet_frames["left_foot"]: [initial_left_footstep], feet_frames["right_foot"]: [initial_right_footstep]}

        return StorageHandler(footsteps_path,
                              postural_path,
                              joystick_input_path,
                              blending_coefficients_path,
                              feet_frames=feet_frames,
                              initial_l_foot_position=initial_l_foot_position,
                              initial_r_foot_position=initial_r_foot_position,
                              footsteps=footsteps,
                              generation_to_control_time_scaling=generation_to_control_time_scaling,
                              time_scaling=time_scaling)

    def update_joystick_inputs_storage(self, raw_data: List, quad_bezier: List, base_velocities: List, facing_dirs: List) -> None:
        """Update the storage of the joystick inputs."""

        # Replicate the joystick inputs at the desired frequency
        for _ in range(self.generation_to_control_time_scaling * self.time_scaling):

            self.joystick_inputs["raw_data"].append(raw_data)
            self.joystick_inputs["quad_bezier"].append(quad_bezier)
            self.joystick_inputs["base_velocities"].append(base_velocities)
            self.joystick_inputs["facing_dirs"].append(facing_dirs)

    def update_blending_coefficients_storage(self, blending_coefficients: List) -> None:
        """Update the storage of the blending coefficients."""

        # Replicate the blending coefficients at the desired frequency
        for _ in range(self.generation_to_control_time_scaling * self.time_scaling):

            self.blending_coeffs["w_1"].append(float(blending_coefficients[0][0]))
            self.blending_coeffs["w_2"].append(float(blending_coefficients[0][1]))
            self.blending_coeffs["w_3"].append(float(blending_coefficients[0][2]))
            self.blending_coeffs["w_4"].append(float(blending_coefficients[0][3]))

    def update_footsteps_storage(self, support_foot: str, footstep: Dict) -> None:
        """Add a footstep to the footsteps storage."""

        self.footsteps[support_foot].append(footstep)

    # TODO: remove?
    def replace_footsteps_storage(self, footsteps: Dict) -> None:
        """Replace the storage of footsteps with an updated footsteps list."""

        self.footsteps = footsteps

    def retrieve_smoothed_dict(self, prev: Dict, next: Dict, steps: int) -> List:
        """Compute a list of dictionaries which smoothly transition from the previous dictionary to the next one.
        The list contains as many dictionaries as the specified 'steps' parameter."""

        # Compute how much each entry of the dictionaries should be updated at each smoothing step
        update = {}
        for key in prev:
            update[key] = (next[key] - prev[key]) / steps

        # Fill the smoothed list of dictionaries by iteratively adding the update to the previous dictionary
        smoothed_list = []
        curr = prev.copy()
        for _ in range(steps-1):
            for key in curr:
                curr[key] += update[key]
            smoothed_list.append(curr)

        # Add the next dictionary as last element of the smoothed list
        smoothed_list.append(next)

        return smoothed_list

    def retrieve_smoothed_3Dsignal(self, prev: List, next: List, steps: int) -> List:
        """Compute a list of 3D signals which smoothly transition from the previous 3D signal to the next one.
        The list contains as many 3D signals as the specified 'steps' parameter."""

        # Compute how much the 3D signal should be updated at each smoothing step
        update = (np.array(next) - np.array(prev)) / steps

        # Fill the smoothed list of 3D signals by iteratively adding the update to the previous 3D signal
        smoothed_list = []
        curr = prev.copy()
        for _ in range(steps-1):
            curr += update
            smoothed_list.append(list(curr))

        # Add the next 3D signal as last element of the smoothed list
        smoothed_list.append(next)

        return smoothed_list

    def update_posturals_storage(self, base: Dict, joints_pos: Dict, joints_vel: Dict,
                                 links: Dict, com_pos: List, com_vel: List, centroidal_momentum: List) -> None:
        """Update the storage of the posturals."""

        # ======
        # JOINTS
        # ======

        # Scale the joint velocities according to the time scaling of the generated trajectory
        joints_vel_scaled = {joint: 1/self.time_scaling * joints_vel[joint] for joint in joints_vel}

        if self.posturals["joints_pos"] == []:

            # Replicate the joints postural at the first iteration
            for _ in range(self.generation_to_control_time_scaling * self.time_scaling):
                self.posturals["joints_pos"].append(joints_pos)
                self.posturals["joints_vel"].append(joints_vel_scaled)

        else:

            # Smooth the joints postural at the desired frequency
            joints_pos_prev = self.posturals["joints_pos"][-1]
            joints_vel_prev = self.posturals["joints_vel"][-1]
            smoothed_joints_pos = self.retrieve_smoothed_dict(joints_pos_prev, joints_pos, self.generation_to_control_time_scaling * self.time_scaling)
            smoothed_joints_vel = self.retrieve_smoothed_dict(joints_vel_prev, joints_vel_scaled, self.generation_to_control_time_scaling * self.time_scaling)
            self.posturals["joints_pos"].extend(smoothed_joints_pos)
            self.posturals["joints_vel"].extend(smoothed_joints_vel)

        # ====
        # BASE
        # ====

        if self.posturals["base"] == []:

            # Replicate the base at the first iteration
            for _ in range(self.generation_to_control_time_scaling * self.time_scaling):
                self.posturals["base"].append(base)

        else:

            # Smooth the base position at the desired frequency
            base_pos_prev = self.posturals["base"][-1]["position"].copy()
            smoothed_base_pos = self.retrieve_smoothed_3Dsignal(base_pos_prev, base["position"], self.generation_to_control_time_scaling * self.time_scaling)

            # Replicate the base orientation at the desired frequency # TODO: slerp on quaternions ?
            smoothed_base_quat = []
            for _ in range(self.generation_to_control_time_scaling * self.time_scaling):
                smoothed_base_quat.append(base["wxyz_quaternions"])

            # Fill the smoothed base postural
            for i in range(len(smoothed_base_quat)):
                smoothed_base = {"position": smoothed_base_pos[i] , "wxyz_quaternions" : smoothed_base_quat[i]}
                self.posturals["base"].extend([smoothed_base])

        # ===
        # COM
        # ===

        # Scale the CoM velocity according to the time scaling of the generated trajectory
        com_vel_scaled = list(1/self.time_scaling * np.array(com_vel))

        if self.posturals["com_pos"] == []:

            # Replicate the com at the first iteration
            for _ in range(self.generation_to_control_time_scaling * self.time_scaling):
                self.posturals["com_pos"].append(com_pos)
                self.posturals["com_vel"].append(com_vel_scaled)

        else:

            # Smooth the com at the desired frequency
            com_pos_prev = self.posturals["com_pos"][-1].copy()
            com_vel_prev = self.posturals["com_vel"][-1].copy()
            smoothed_com_pos = self.retrieve_smoothed_3Dsignal(com_pos_prev, com_pos, self.generation_to_control_time_scaling * self.time_scaling)
            smoothed_com_vel = self.retrieve_smoothed_3Dsignal(com_vel_prev, com_vel_scaled, self.generation_to_control_time_scaling * self.time_scaling)
            self.posturals["com_pos"].extend(smoothed_com_pos)
            self.posturals["com_vel"].extend(smoothed_com_vel)

        # ===================
        # CENTROIDAL MOMENTUM
        # ===================

        # Scale the centroidal momentum according to the time scaling of the generated trajectory
        centroidal_momentum_scaled = [list(1/self.time_scaling * np.array(centroidal_momentum[0])),
                                      list(1/self.time_scaling * np.array(centroidal_momentum[1]))]

        if self.posturals["centroidal_momentum"] == []:

            # Replicate the centroidal_momentum at the first iteration
            for _ in range(self.generation_to_control_time_scaling * self.time_scaling):
                self.posturals["centroidal_momentum"].append(centroidal_momentum_scaled)

        else:

            # Retrieve linear and angular momentum
            centroidal_momentum_prev = self.posturals["centroidal_momentum"][-1].copy()
            linear_momentum_prev = centroidal_momentum_prev[0]
            angular_momentum_prev = centroidal_momentum_prev[1]
            linear_momentum_next = centroidal_momentum_scaled[0]
            angular_momentum_next = centroidal_momentum_scaled[1]

            # Smooth linear and angular momentum at the desired frequency
            smoothed_linear_momentum = self.retrieve_smoothed_3Dsignal(linear_momentum_prev, linear_momentum_next, self.generation_to_control_time_scaling * self.time_scaling)
            smoothed_angular_momentum = self.retrieve_smoothed_3Dsignal(angular_momentum_prev, angular_momentum_next, self.generation_to_control_time_scaling * self.time_scaling)

            # Retrieve smoothed centroidal momentum
            smoothed_centroidal_momentum = [[smoothed_linear_momentum[k], smoothed_angular_momentum[k]] for k in range(len(smoothed_linear_momentum))]
            self.posturals["centroidal_momentum"].extend(smoothed_centroidal_momentum)

    # TODO: integrate once we define the message to be passed via YARP
    # def retrieve_contacts(self, plot_contacts: bool = False) -> blf.contacts.ContactPhaseList:
    #     """Retrieve planned contacts from the generated trajectory, and optionally plot them."""
    #
    #     # Footsteps list
    #     contact_phase_list: blf.contacts.ContactPhaseList
    #
    #     # Create the map of contact lists
    #     contact_list_map = dict()
    #     contact_list_map[self.feet_frames["left_foot"]] = blf.contacts.ContactList()
    #     contact_list_map[self.feet_frames["right_foot"]] = blf.contacts.ContactList()
    #
    #     # Retrieve footsteps
    #     l_contacts = self.footsteps[self.feet_frames["left_foot"]]
    #     r_contacts = self.footsteps[self.feet_frames["right_foot"]]
    #
    #     if plot_contacts:
    #         # Storage fot plotting
    #         left_footsteps_x = []
    #         left_footsteps_y = []
    #         right_footsteps_x = []
    #         right_footsteps_y = []
    #
    #     # ================
    #     # INITIAL CONTACTS
    #     # ================
    #
    #     # Retrieve first left contact position # TODO: remove and rename
    #     ground_l_foot_position = l_contacts[0]["pos"]
    #     ground_l_foot_position_offset = np.array(self.initial_l_foot_position) - np.array(ground_l_foot_position)
    #     ground_l_foot_position += np.array(ground_l_foot_position_offset)
    #
    #     # Retrieve first left contact orientation
    #     l_foot_quat = l_contacts[0]["quat"]
    #     l_deactivation_time = self.time_scaling * (l_contacts[0]["deactivation_time"])
    #
    #     # Retrieve first right contact position # TODO: remove and rename
    #     ground_r_foot_position = r_contacts[0]["pos"]
    #     ground_r_foot_position_offset = np.array(self.initial_r_foot_position) - np.array(ground_r_foot_position)
    #     ground_r_foot_position += np.array(ground_r_foot_position_offset)
    #
    #     # Retrieve first right contact orientation
    #     r_foot_quat = r_contacts[0]["quat"]
    #     r_deactivation_time = self.time_scaling * (r_contacts[0]["deactivation_time"])
    #
    #     # Add initial left and right contacts to the list
    #     assert contact_list_map[self.feet_frames["left_foot"]].add_contact(
    #         transform=manif.SE3(position=np.array(ground_l_foot_position),
    #                             quaternion=l_foot_quat),
    #         activation_time=0.0,
    #         deactivation_time=l_deactivation_time)
    #     assert contact_list_map[self.feet_frames["right_foot"]].add_contact(
    #         transform=manif.SE3(position=np.array(ground_r_foot_position),
    #                             quaternion=r_foot_quat),
    #         activation_time=0.0,
    #         deactivation_time=r_deactivation_time)
    #
    #     # =============
    #     # LEFT CONTACTS
    #     # =============
    #
    #     if plot_contacts:
    #         # Update storage for plotting
    #         left_footsteps_x.append(ground_l_foot_position[0])
    #         left_footsteps_y.append(ground_l_foot_position[1])
    #
    #     for contact in l_contacts[1:]:
    #
    #         # Retrieve position
    #         ground_l_foot_position = contact["pos"]
    #         ground_l_foot_position += np.array(ground_l_foot_position_offset) # TODO: remove
    #
    #         if plot_contacts:
    #             # Update storage for plotting
    #             left_footsteps_x.append(ground_l_foot_position[0])
    #             left_footsteps_y.append(ground_l_foot_position[1])
    #
    #         # Retrieve orientation and timing
    #         l_foot_quat = contact["quat"]
    #         l_activation_time = self.time_scaling * (contact["activation_time"])
    #         l_deactivation_time = self.time_scaling * (contact["deactivation_time"])
    #
    #         # Add the contact
    #         assert contact_list_map[self.feet_frames["left_foot"]].add_contact(
    #             transform=manif.SE3(position=np.array(ground_l_foot_position), quaternion=l_foot_quat),
    #             activation_time=l_activation_time,
    #             deactivation_time=l_deactivation_time)
    #
    #     # ==============
    #     # RIGHT CONTACTS
    #     # ==============
    #
    #     if plot_contacts:
    #         # Update storage for plotting
    #         right_footsteps_x.append(ground_r_foot_position[0])
    #         right_footsteps_y.append(ground_r_foot_position[1])
    #
    #     for contact in r_contacts[1:]:
    #
    #         # Retrieve position
    #         ground_r_foot_position = contact["pos"]
    #         ground_r_foot_position += np.array(ground_r_foot_position_offset)  # TODO: remove
    #
    #         if plot_contacts:
    #             # Update storage for plotting
    #             right_footsteps_x.append(ground_r_foot_position[0])
    #             right_footsteps_y.append(ground_r_foot_position[1])
    #
    #         # Retrieve orientation and timing
    #         r_foot_quat = contact["quat"]
    #         r_activation_time = self.time_scaling * (contact["activation_time"])
    #         r_deactivation_time = self.time_scaling * (contact["deactivation_time"])
    #
    #         # Add the contact
    #         assert contact_list_map[self.feet_frames["right_foot"]].add_contact(
    #             transform=manif.SE3(position=np.array(ground_r_foot_position), quaternion=r_foot_quat),
    #             activation_time=r_activation_time,
    #             deactivation_time=r_deactivation_time)
    #
    #     # ====
    #     # PLOT
    #     # ====
    #
    #     if plot_contacts:
    #         # Plot contacts
    #         plt.figure()
    #         plt.plot(left_footsteps_x, left_footsteps_y, 'b', label='left contacts')
    #         plt.plot(right_footsteps_x, right_footsteps_y, 'r', label='right contacts')
    #         plt.scatter(left_footsteps_x, left_footsteps_y, c='b')
    #         plt.scatter(right_footsteps_x, right_footsteps_y, c='r')
    #         plt.legend()
    #         plt.title("Contacts")
    #         plt.axis("equal")
    #         plt.show(block=False)
    #         plt.pause(0.5)
    #
    #     # Assign contact list
    #     contact_phase_list = blf.contacts.ContactPhaseList()
    #     contact_phase_list.set_lists(contact_lists=contact_list_map)
    #
    #     return contact_phase_list

    # TODO: remove
    def save_data_as_json(self) -> None:
        """Save all the stored data using the json format."""

        # Save footsteps
        with open(self.footsteps_path, 'w') as outfile:
            json.dump(self.footsteps, outfile)

        # Save postural
        with open(self.postural_path, 'w') as outfile:
            json.dump(self.posturals, outfile)

        # Save joystick inputs
        with open(self.joystick_input_path, 'w') as outfile:
            json.dump(self.joystick_inputs, outfile)

        # Save blending coefficients
        with open(self.blending_coefficients_path, 'w') as outfile:
            json.dump(self.blending_coeffs, outfile)

        # Debug
        input("\nData have been saved. Press Enter to continue the trajectory generation.")

    def update_mergepoint_state(self) -> None:
        """Update the storage of footsteps and postural at the last mergepoint."""

        # Footsteps
        self.mergepoint_footsteps = {}
        for foot in self.footsteps.keys():
            self.mergepoint_footsteps[foot] = []
            for footstep in self.footsteps[foot]:
                self.mergepoint_footsteps[foot].append(footstep.copy())

        # Postural # TODO
        # self.mergepoint_posturals = self.posturals.copy()

    def reset_from_mergepoint(self, updated_activation_time: int) -> None:
        """Update the footsteps and postural from the last mergepoint."""

        # Footsteps
        for foot in self.footsteps.keys():
            last_mergepoint_footstep = self.mergepoint_footsteps[foot][-1].copy()
            last_mergepoint_footstep["activation_time"] = updated_activation_time
            self.footsteps[foot] = [last_mergepoint_footstep]

        # Postural TODO
        for key in self.posturals.keys():
            self.posturals[key] = []


@dataclass
class FootstepsExtractor:
    """Class to extract the footsteps from the generated trajectory."""

    # Define robot-specific feet frames definition
    feet_frames: Dict

    # Time scaling factor for the generated trajectory
    time_scaling: int

    # Auxiliary variables for the footsteps update before saving
    nominal_DS_duration: float
    difference_position_threshold: float
    difference_time_threshold: float

    # Auxiliary variables to handle the footsteps deactivation time
    difference_height_norm_threshold: bool
    waiting_for_deactivation_time: bool = True

    @staticmethod
    def build(feet_frames: Dict,
              time_scaling: int = 1,
              nominal_DS_duration: float = 0.04,
              difference_position_threshold: float = 0.04,
              difference_time_threshold: float = 0.10, # TODO: tune
              difference_height_norm_threshold: bool = 0.005) -> "FootstepsExtractor":
        """Build an instance of FootstepsExtractor."""

        return FootstepsExtractor(feet_frames=feet_frames,
                                  nominal_DS_duration=nominal_DS_duration,
                                  difference_position_threshold=difference_position_threshold,
                                  difference_time_threshold=difference_time_threshold,
                                  difference_height_norm_threshold=difference_height_norm_threshold,
                                  time_scaling=time_scaling)

    def should_update_footstep_deactivation_time(self, kindyn: kindyncomputations.KinDynComputations) -> bool:
        """Check whether the deactivation time of the last footstep needs to be updated."""

        # Retrieve the transformation from the world frame to the base frame
        world_H_base = kindyn.get_world_base_transform()

        # Compute right foot height
        base_H_r_foot = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.feet_frames["right_foot"])
        W_H_RF = world_H_base.dot(base_H_r_foot)
        W_right_foot_pos = W_H_RF [0:3, -1]
        right_foot_height = W_right_foot_pos[2]

        # Compute left foot height
        base_H_l_foot = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.feet_frames["left_foot"])
        W_H_LF = world_H_base.dot(base_H_l_foot)
        W_left_foot_pos = W_H_LF[0:3, -1]
        left_foot_height = W_left_foot_pos[2]

        # Compute the difference in height between the feet
        difference_height_norm = np.linalg.norm(left_foot_height - right_foot_height)

        # If the height difference is above a threshold and a foot is being detached, the deactivation
        # time of the last footstep related to the detaching foot needs to be updated
        if self.waiting_for_deactivation_time and difference_height_norm > self.difference_height_norm_threshold:
            self.waiting_for_deactivation_time = False
            return True

        return False

    def create_new_footstep(self, kindyn: kindyncomputations.KinDynComputations,
                            support_foot: str, activation_time: float) -> Dict:
        """Retrieve the information related to a new footstep."""

        new_footstep = {}

        # Compute new footstep position
        world_H_base = kindyn.get_world_base_transform()
        base_H_support_foot = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=support_foot)
        W_H_SF = world_H_base.dot(base_H_support_foot)
        support_foot_pos = W_H_SF[0:3, -1]
        new_footstep["pos"] = list(support_foot_pos)

        # Compute new footstep orientation
        new_footstep["quat"] = list(Quaternion.from_matrix(W_H_SF[0:3, 0:3]))

        # Assign new footstep activation time
        new_footstep["activation_time"] = activation_time

        # Use a temporary flag indicating that the deactivation time has not been computed yet
        new_footstep["deactivation_time"] = -1

        # Set the flag indicating that the last footstep has no deactivation time yet accordingly
        self.waiting_for_deactivation_time = True

        return new_footstep

    # TODO: remove (and do it online if needed)
    def update_footsteps(self, final_deactivation_time: float, footsteps: Dict) -> Dict:
        """Update the footsteps list before saving data by replacing temporary deactivation times (if any) and
        merging footsteps which are too close each other in order to avoid unintended footsteps on the spot.
        """

        # Update the deactivation time of the last footstep of each foot (they need to coincide to be processed
        # properly in the trajectory control layer)
        for foot in footsteps.keys():
            footsteps[foot][-1]["deactivation_time"] = self.time_scaling * final_deactivation_time
        # Replace temporary deactivation times in the footsteps list (if any)
        updated_footsteps = self.replace_temporary_deactivation_times(footsteps=footsteps)

        # Merge footsteps which are too close each other
        updated_footsteps = self.merge_close_footsteps(final_deactivation_time=final_deactivation_time,
                                                       footsteps=updated_footsteps)

        return updated_footsteps

    # TODO: remove (and do it online if needed)
    def replace_temporary_deactivation_times(self, footsteps: Dict) -> Dict:
        """Replace temporary footstep deactivation times that may not have been updated properly."""

        # Map from one foot to the other
        other_foot = {self.feet_frames["left_foot"]: self.feet_frames["right_foot"],
                      self.feet_frames["right_foot"]: self.feet_frames["left_foot"]}

        for foot in [self.feet_frames["left_foot"],self.feet_frames["right_foot"]]:

            for footstep in footsteps[foot]:

                # If a temporary footstep deactivation time is detected
                if footstep["deactivation_time"] == -1:

                    # Retrieve the footstep activation time
                    current_activation_time = footstep["activation_time"]

                    for footstep_other_foot in footsteps[other_foot[foot]]:

                        # Retrieve the activation time of the next footstep of the other foot
                        other_foot_activation_time = footstep_other_foot["activation_time"]

                        if other_foot_activation_time > current_activation_time:

                            # Update the deactivation time so to have a double support (DS) phase of the nominal duration
                            current_deactivation_time = other_foot_activation_time + \
                                                        self.time_scaling * self.nominal_DS_duration
                            footstep["deactivation_time"] = current_deactivation_time

                            break

        return footsteps

    # TODO: remove (and do it online if needed)
    def merge_close_footsteps(self, final_deactivation_time: float, footsteps: Dict) -> Dict:
        """Merge footsteps that are too close each other (in terms of space and time) to avoid unintended footsteps on the spot."""

        # Initialize updated footsteps list
        updated_footsteps = {self.feet_frames["left_foot"]: [], self.feet_frames["right_foot"]: []}

        for foot in footsteps.keys():

            # Auxiliary variable to handle footsteps update
            skip_next_contact = False

            for i in range(len(footsteps[foot]) - 1):

                if skip_next_contact:
                    skip_next_contact = False
                    continue

                # Compute the norm of the difference in position between consecutive footsteps of the same foot
                current_footstep_position = np.array(footsteps[foot][i]["pos"])
                next_footstep_position = np.array(footsteps[foot][i + 1]["pos"])
                difference_position = np.linalg.norm(current_footstep_position - next_footstep_position)

                # Compute the difference in time between consecutive footsteps of the same foot
                current_footstep_deactivation = np.array(footsteps[foot][i]["deactivation_time"])
                next_footstep_activation = np.array(footsteps[foot][i + 1]["activation_time"])
                difference_time = next_footstep_activation - current_footstep_deactivation

                if difference_position >= self.difference_position_threshold or \
                        (difference_position < self.difference_position_threshold and
                         difference_time >= self.time_scaling * self.difference_time_threshold):

                    # Do not update footsteps which are not enough close each other (in terms of both time and space)
                    updated_footsteps[foot].append(footsteps[foot][i])

                else:

                    # Merge footsteps which are close each other: the duration of the current footstep is extended
                    # till the end of the subsequent footstep
                    updated_footstep = footsteps[foot][i]
                    updated_footstep["deactivation_time"] = footsteps[foot][i + 1]["deactivation_time"]
                    updated_footsteps[foot].append(updated_footstep)
                    skip_next_contact = True

            # If the last updated footstep ends before the final deactivation time, add the last original footstep
            # to the updated list of footsteps
            if updated_footsteps[foot]==[] or updated_footsteps[foot][-1]["deactivation_time"] != final_deactivation_time * self.time_scaling:
                updated_footsteps[foot].append(footsteps[foot][-1])

        return updated_footsteps

    def check_close_footsteps(self, current_footstep: Dict, next_footstep: Dict) -> bool:
        """Check whether the current and the next footsteps are too close each other
        (in terms of space and time) to avoid unintended footsteps on the spot."""

        # TODO: prev foorstep need to be of the same foot! Where is this guaranteed?

        close_footsteps = False

        # Compute the norm of the difference in position between current and next footsteps
        current_footstep_position = np.array(current_footstep["pos"])
        next_footstep_position = np.array(next_footstep["pos"])
        difference_position = np.linalg.norm(current_footstep_position - next_footstep_position)
        # Debug # TODO: remove
        # print("current_footstep_position: ", current_footstep_position)
        # print("next_footstep_position: ", next_footstep_position)
        # print("difference position: ", difference_position)

        # Compute the norm of the difference in time between current and next footsteps
        current_footstep_deactivation = np.array(current_footstep["deactivation_time"])
        next_footstep_activation = np.array(next_footstep["activation_time"])
        difference_time = next_footstep_activation - current_footstep_deactivation
        # Debug # TODO: remove
        # print("current_footstep_deactivation: ", current_footstep_deactivation)
        # print("next_footstep_activation: ", next_footstep_activation)
        # print("difference_time: ", difference_time)# TODO: is this wrong because of the -1?

        # Check whether the footsteps are close in position OR in time
        if difference_position < self.difference_position_threshold or \
                (current_footstep_deactivation != -1 and
                 difference_time < self.time_scaling * self.difference_time_threshold):
        # TODO: possibly check whether the footsteps are close in position AND in time ?
        # if difference_position < self.difference_position_threshold and \
        #         (current_footstep_deactivation == -1 or
        #          difference_time < self.time_scaling * self.difference_time_threshold):

            close_footsteps = True

        return close_footsteps


@dataclass
class PosturalExtractor:
    """Class to extract several posturals from the generated trajectory."""

    @staticmethod
    def build() -> "PosturalExtractor":
        """Build an instance of PosturalExtractor."""

        return PosturalExtractor()

    @staticmethod
    def create_new_posturals(base_position: List, base_quaternion: List, joint_positions: List, controlled_joints: List,
                             kindyn: kindyncomputations.KinDynComputations, link_names: List) -> (List, List, List, List):
        """Retrieve the information related to a new set of postural terms."""

        # Store the postural term related to the base position and orientation
        new_base_postural = {"position": list(base_position), "wxyz_quaternions": list(base_quaternion)}

        # Store the postural term related to the joint angles
        new_joints_pos_postural = {controlled_joints[k]: joint_positions[k] for k in range(len(controlled_joints))}

        # Store the postural term related to the joint velocities
        joint_velocities = kindyn.get_joint_velocities()
        new_joints_vel_postural = {controlled_joints[k]: joint_velocities[k] for k in range(len(controlled_joints))}

        # Store the postural term related to the link orientations
        new_links_postural = {}
        world_H_base = kindyn.get_world_base_transform()
        for link_name in link_names:
            base_H_link = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=link_name)
            world_H_link = world_H_base.dot(base_H_link)
            new_links_postural[link_name] = list(Quaternion.from_matrix(world_H_link[0:3, 0:3]))

        # Store the postural term related to the com positions
        new_com_pos_postural = list(kindyn.get_com_position())

        # Store the postural term related to the com velocities
        new_com_vel_postural = list(kindyn.get_com_velocity())

        # Store the postural term related to the centroidal momentum
        centroidal_momentum = list(kindyn.get_centroidal_momentum())
        new_centroidal_momentum_postural = [list(centroidal_momentum[0]),list(centroidal_momentum[1])]

        return new_base_postural, new_joints_pos_postural, new_joints_vel_postural, new_links_postural, \
               new_com_pos_postural, new_com_vel_postural, new_centroidal_momentum_postural


@dataclass
class KinematicComputations:
    """Class for the kinematic computations exploited within the trajectory generation pipeline to compute
    kinematically-feasible base motions.
    """

    kindyn: kindyncomputations.KinDynComputations

    # Footsteps and postural extractors
    footsteps_extractor: FootstepsExtractor
    postural_extractor: PosturalExtractor

    # Simulated robot (for visualization only)
    icub: iCub
    controlled_joints: List
    controlled_joints_indexes: List
    gazebo: scenario.GazeboSimulator

    # Kinematics-related quantities at the current timestep
    local_foot_vertices_pos: List
    feet_frames: Dict
    feet_links: Dict
    support_foot_prev: str
    support_foot: str
    support_vertex_prev: int
    support_vertex: int
    support_foot_pos: float = 0
    support_vertex_pos: float = 0
    support_vertex_offset: float = 0
    joint_positions: List = None
    joint_velocities: List = None
    base_position: List = None
    base_quaternion: List = None

    # Kinematics-related quantities at the last mergepoint
    mergepoint_support_foot_prev: str = ""
    mergepoint_support_foot: str = ""
    mergepoint_support_vertex_prev: int = 0
    mergepoint_support_vertex: int = 0
    mergepoint_support_foot_pos: float = 0
    mergepoint_support_vertex_pos: float = 0
    mergepoint_support_vertex_offset: float = 0
    mergepoint_joint_positions: List = None
    mergepoint_joint_velocities: List = None
    mergepoint_base_position: List = None
    mergepoint_base_quaternion: List = None

    @staticmethod
    def build(kindyn: kindyncomputations.KinDynComputations,
              controlled_joints_indexes: List,
              local_foot_vertices_pos: List,
              feet_frames: Dict,
              feet_links: Dict,
              icub: iCub,
              gazebo: scenario.GazeboSimulator,
              initial_support_foot: str,
              initial_support_vertex: int,
              time_scaling: int,
              nominal_DS_duration: float = 0.04,
              difference_position_threshold: float = 0.04,
              difference_height_norm_threshold: bool = 0.005) -> "KinematicComputations":
        """Build an instance of KinematicComputations."""

        footsteps_extractor = FootstepsExtractor.build(feet_frames=feet_frames,
                                                       nominal_DS_duration=nominal_DS_duration,
                                                       difference_position_threshold=difference_position_threshold,
                                                       difference_height_norm_threshold=difference_height_norm_threshold,
                                                       time_scaling=time_scaling)

        postural_extractor = PosturalExtractor.build()

        return KinematicComputations(kindyn=kindyn,
                                     controlled_joints_indexes=controlled_joints_indexes,
                                     footsteps_extractor=footsteps_extractor,
                                     postural_extractor=postural_extractor,
                                     local_foot_vertices_pos=local_foot_vertices_pos,
                                     feet_frames=feet_frames,
                                     feet_links=feet_links,
                                     support_foot_prev=feet_frames[initial_support_foot],
                                     support_foot=feet_frames[initial_support_foot],
                                     support_vertex_prev=initial_support_vertex,
                                     support_vertex=initial_support_vertex,
                                     icub=icub,
                                     controlled_joints=icub.joint_names(),
                                     gazebo=gazebo)

    def compute_W_vertices_pos(self) -> List:
        """Compute the feet vertices positions in the world (W) frame."""

        # Retrieve the transformation from the world to the base frame
        world_H_base = self.kindyn.get_world_base_transform()

        # Retrieve front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = self.local_foot_vertices_pos[0]
        FR_vertex_pos = self.local_foot_vertices_pos[1]
        BL_vertex_pos = self.local_foot_vertices_pos[2]
        BR_vertex_pos = self.local_foot_vertices_pos[3]

        # Compute right foot (RF) transform w.r.t. the world (W) frame
        base_H_r_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.feet_frames["right_foot"])
        W_H_RF = world_H_base.dot(base_H_r_foot)

        # Get the right-foot vertices positions in the world frame
        W_RFL_vertex_pos_hom = W_H_RF @ np.concatenate((FL_vertex_pos, [1]))
        W_RFR_vertex_pos_hom = W_H_RF @ np.concatenate((FR_vertex_pos, [1]))
        W_RBL_vertex_pos_hom = W_H_RF @ np.concatenate((BL_vertex_pos, [1]))
        W_RBR_vertex_pos_hom = W_H_RF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_RFL_vertex_pos = W_RFL_vertex_pos_hom[0:3]
        W_RFR_vertex_pos = W_RFR_vertex_pos_hom[0:3]
        W_RBL_vertex_pos = W_RBL_vertex_pos_hom[0:3]
        W_RBR_vertex_pos = W_RBR_vertex_pos_hom[0:3]

        # Compute left foot (LF) transform w.r.t. the world (W) frame
        base_H_l_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.feet_frames["left_foot"])
        W_H_LF = world_H_base.dot(base_H_l_foot)

        # Get the left-foot vertices positions wrt the world frame
        W_LFL_vertex_pos_hom = W_H_LF @ np.concatenate((FL_vertex_pos, [1]))
        W_LFR_vertex_pos_hom = W_H_LF @ np.concatenate((FR_vertex_pos, [1]))
        W_LBL_vertex_pos_hom = W_H_LF @ np.concatenate((BL_vertex_pos, [1]))
        W_LBR_vertex_pos_hom = W_H_LF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_LFL_vertex_pos = W_LFL_vertex_pos_hom[0:3]
        W_LFR_vertex_pos = W_LFR_vertex_pos_hom[0:3]
        W_LBL_vertex_pos = W_LBL_vertex_pos_hom[0:3]
        W_LBR_vertex_pos = W_LBR_vertex_pos_hom[0:3]

        # Store the positions of both right-foot and left-foot vertices in the world frame
        W_vertices_positions = [W_RFL_vertex_pos, W_RFR_vertex_pos, W_RBL_vertex_pos, W_RBR_vertex_pos,
                                W_LFL_vertex_pos, W_LFR_vertex_pos, W_LBL_vertex_pos, W_LBR_vertex_pos]

        return W_vertices_positions

    def set_initial_support_vertex_and_support_foot(self) -> None:
        """Compute initial support foot and support vertex positions in the world frame, along with the support vertex offset."""

        # Compute support foot position wrt the world frame
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_SF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
        W_H_SF = world_H_base.dot(base_H_SF)
        W_support_foot_pos = W_H_SF[0:3, -1]

        # Compute support vertex position wrt the world frame
        F_support_vertex_pos = self.local_foot_vertices_pos[self.support_vertex % 4]
        F_support_vertex_pos_hom = np.concatenate((F_support_vertex_pos, [1]))
        W_support_vertex_pos_hom = W_H_SF @ F_support_vertex_pos_hom
        W_support_vertex_pos = W_support_vertex_pos_hom[0:3]

        # Set initial support foot and support vertex positions, along with the support vertex offset
        self.support_foot_pos = W_support_foot_pos
        self.support_vertex_pos = W_support_vertex_pos
        self.support_vertex_offset = [W_support_vertex_pos[0], W_support_vertex_pos[1], 0]

    def reset_robot_configuration(self,
                                  joint_positions: List,
                                  joint_velocities: List,
                                  base_position: List,
                                  base_quaternion: List) -> None:
        """Reset the robot configuration."""

        # Retrieve the transformation from the world frame to the base frame
        world_H_base = numpy.FromNumPy.to_idyntree_transform(
            position=np.array(base_position),
            quaternion=np.array(base_quaternion)).asHomogeneousTransform().toNumPy()

        # Extract the base velocity by imposing the holonomic constraint on the current support foot
        jacobian_SF = self.kindyn.get_frame_jacobian(self.feet_links[self.support_foot])
        jacobian_SF_base = jacobian_SF[:, :6]
        jacobian_SF_joints = jacobian_SF[:, 6:]
        base_velocity = - np.linalg.inv(jacobian_SF_base).dot(jacobian_SF_joints.dot(joint_velocities))

        # Set the robot state using the base velocity retrieved by legged odometry
        self.kindyn.set_robot_state(s=joint_positions, ds=joint_velocities, world_H_base=world_H_base, base_velocity=base_velocity)

        # Store current kinematics-related quantities
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.base_position = base_position
        self.base_quaternion = base_quaternion

    def reset_visual_robot_configuration(self,
                                         joint_positions: List = None,
                                         base_position: List = None,
                                         base_quaternion: List = None) -> None:
        """Reset the configuration of the robot visualized in the simulator."""

        # Reset joint configuration
        if joint_positions is not None:
            full_joint_positions = np.zeros(len(self.controlled_joints))
            for i in range(len(joint_positions)):
                full_joint_positions[self.controlled_joints_indexes[i]] = joint_positions[i]
            self.icub.to_gazebo().reset_joint_positions(full_joint_positions, self.controlled_joints)

        # Reset base pose
        if base_position is not None and base_quaternion is not None:
            self.icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        elif base_quaternion is not None:
            self.icub.to_gazebo().reset_base_orientation(base_quaternion)
        elif base_position is not None:
            self.icub.to_gazebo().reset_base_position(base_position)

        # Step the simulator (visualization only)
        self.gazebo.run(paused=True)

    def compute_and_apply_kinematically_feasible_base_position(self,
                                                               joint_positions: List,
                                                               joint_velocities: List,
                                                               base_quaternion: List) -> List:
        """Compute kinematically-feasible base position and update the robot configuration."""

        # Recompute base position by leg odometry
        kinematically_feasible_base_pos = self.compute_base_position_by_leg_odometry()

        # Update the base position in the robot configuration
        self.reset_robot_configuration(joint_positions=joint_positions,
                                       joint_velocities=joint_velocities,
                                       base_position=kinematically_feasible_base_pos,
                                       base_quaternion=base_quaternion)

        # Update the base position in the configuration of the robot visualized in the simulator
        self.reset_visual_robot_configuration(base_position=kinematically_feasible_base_pos)

        return kinematically_feasible_base_pos

    def compute_base_position_by_leg_odometry(self) -> List:
        """Compute kinematically-feasible base position using leg odometry."""

        # Get the base (B) position in the world (W) frame
        W_pos_B = self.kindyn.get_world_base_transform()[0:3, -1]

        # Get the support vertex position in the world (W) frame
        W_support_vertex_pos = self.support_vertex_pos

        # Get the support vertex orientation in the world (W) frame, defined as the support foot (SF) orientation
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_SF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
        W_H_SF = world_H_base.dot(base_H_SF)
        W_support_vertex_quat = Quaternion.from_matrix(W_H_SF[0:3, 0:3])

        # Compute the transform of the support vertex (SV) in the world (W) frame
        W_H_SV = Transform.from_position_and_quaternion(position=np.asarray(W_support_vertex_pos),
                                                        quaternion=np.asarray(W_support_vertex_quat))

        # Express the base (B) position in the support vertex (SV) reference frame
        SV_H_W = np.linalg.inv(W_H_SV)
        W_pos_B_hom = np.concatenate((W_pos_B, [1]))
        SV_pos_B_hom = SV_H_W @ W_pos_B_hom

        # Express the base (B) position in a reference frame oriented as the world but positioned in the support vertex (SV)
        mixed_H_SV = Transform.from_position_and_quaternion(position=np.asarray([0, 0, 0]),
                                                            quaternion=np.asarray(W_support_vertex_quat))
        mixed_pos_B_hom = mixed_H_SV @ SV_pos_B_hom

        # Convert homogeneous to cartesian coordinates
        mixed_pos_B = mixed_pos_B_hom[0:3]

        # Compute the kinematically-feasible base position, i.e. the base position such that the support
        # vertex remains fixed while the robot configuration changes
        kinematically_feasible_base_position = mixed_pos_B + self.support_vertex_offset

        return kinematically_feasible_base_position

    def update_support_vertex_pos(self) -> None:
        """Update the support vertex position."""

        # Retrieve the vertices positions in the world frame
        W_vertices_positions = self.compute_W_vertices_pos()

        # Update the support vertex position
        self.support_vertex_pos = W_vertices_positions[self.support_vertex]

    def update_support_vertex_and_support_foot(self, iteration: int) -> (str, bool, bool):
        """Update the support vertex and the support foot. Also, return boolean variables indicating whether the
        deactivation time of the last footstep needs to be updated (update_footstep_deactivation_time) and whether
        a new footstep needs to be added to the footsteps list (update_footsteps_list)."""

        update_footsteps_list = False

        # Associate feet vertices names to indexes
        vertex_indexes_to_names = {0: "RFL", 1: "RFR", 2: "RBL", 3: "RBR",
                                   4: "LFL", 5: "LFR", 6: "LBL", 7: "LBR"}

        # Retrieve the vertices positions in the world frame
        W_vertices_positions = self.compute_W_vertices_pos()

        # TODO: remove this temporary fix
        # if iteration == 5:
        #     # After a few iterations, force the current support vertex to switch to the other foot (so that you do not miss the initial contacts)
        #     self.support_vertex = (self.support_vertex + 4) % 8
        #     print("iteration:", iteration, "support_vertex:", self.support_vertex)
        # else:
        #     # Compute the current support vertex
        #     vertices_heights = [W_vertex[2] for W_vertex in W_vertices_positions]
        #     self.support_vertex = np.argmin(vertices_heights)

        # Compute the current support vertex
        vertices_heights = [W_vertex[2] for W_vertex in W_vertices_positions]
        self.support_vertex = np.argmin(vertices_heights)

        # Check whether the deactivation time of the last footstep needs to be updated
        update_footstep_deactivation_time = self.footsteps_extractor.should_update_footstep_deactivation_time(kindyn=self.kindyn)

        # Debug # TODO: remove
        # print("UPDATE DEACTIVATION?", update_footstep_deactivation_time)
        # input()

        # If the support vertex did not change
        if self.support_vertex == self.support_vertex_prev:

            # Update support foot position and support vertex position
            world_H_base = self.kindyn.get_world_base_transform()
            base_H_support_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
            W_H_SF = world_H_base.dot(base_H_support_foot)
            self.support_foot_pos = W_H_SF[0:3, -1]
            self.support_vertex_pos = W_vertices_positions[self.support_vertex]

        # If the support vertex changed
        else:

            # Update the support foot
            if vertex_indexes_to_names[self.support_vertex][0] == "R":
                self.support_foot = self.feet_frames["right_foot"]
            else:
                self.support_foot = self.feet_frames["left_foot"]

            # If the support foot changed
            if self.support_foot != self.support_foot_prev:

                # Debug # TODO: remove
                # print("SF changed from ", self.support_foot_prev, " to ", self.support_foot)

                # Indicate that a new footstep needs to be added to the footsteps list
                update_footsteps_list = True

                # Update support foot prev
                self.support_foot_prev = self.support_foot

            # Update support foot position and support vertex position
            world_H_base = self.kindyn.get_world_base_transform()
            base_H_support_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
            W_H_SF = world_H_base.dot(base_H_support_foot)
            self.support_foot_pos = W_H_SF[0:3, -1]
            self.support_vertex_pos = W_vertices_positions[self.support_vertex]

            # Update also the vertex offset
            self.support_vertex_offset = [self.support_vertex_pos[0], self.support_vertex_pos[1], 0]

            # Update support vertex prev
            self.support_vertex_prev = self.support_vertex

        return self.support_foot, update_footstep_deactivation_time, update_footsteps_list

    def update_mergepoint_state(self) -> None:
        """Update the storage of kinematics-related quantities at the last mergepoint."""

        self.mergepoint_support_foot_prev = self.support_foot_prev
        self.mergepoint_support_foot = self.support_foot
        self.mergepoint_support_vertex_prev = self.support_vertex_prev
        self.mergepoint_support_vertex = self.support_vertex
        self.mergepoint_support_foot_pos = self.support_foot_pos
        self.mergepoint_support_vertex_pos = self.support_vertex_pos
        self.mergepoint_support_vertex_offset = self.support_vertex_offset
        self.mergepoint_joint_positions= self.joint_positions
        self.mergepoint_joint_velocities = self.joint_velocities
        self.mergepoint_base_position = self.base_position
        self.mergepoint_base_quaternion = self.base_quaternion

    def reset_from_mergepoint(self) -> None:
        """Reset kinematics-related quantities from the last mergepoint."""

        self.support_foot_prev = self.mergepoint_support_foot_prev
        self.support_foot = self.mergepoint_support_foot
        self.support_vertex_prev = self.mergepoint_support_vertex_prev
        self.support_vertex = self.mergepoint_support_vertex
        self.support_foot_pos = self.mergepoint_support_foot_pos
        self.support_vertex_pos = self.mergepoint_support_vertex_pos
        self.support_vertex_offset = self.mergepoint_support_vertex_offset

        self.reset_robot_configuration(joint_positions=self.mergepoint_joint_positions,
                                       joint_velocities=self.mergepoint_joint_velocities,
                                       base_position=self.mergepoint_base_position,
                                       base_quaternion=self.mergepoint_base_quaternion)
        self.reset_visual_robot_configuration(joint_positions=self.mergepoint_joint_positions,
                                              base_position=self.mergepoint_base_position,
                                              base_quaternion=self.mergepoint_base_quaternion)


@dataclass
class Plotter:
    """Class to handle the plots related to the trajectory generation pipeline."""

    # Define robot-specific feet frames definition
    feet_frames: Dict

    # Define colors used to print the footsteps
    footsteps_colors: Dict

    # Axis of the composed ellipsoid constraining the last point of the Bezier curve of base positions
    ellipsoid_forward_axis: float
    ellipsoid_side_axis: float
    ellipsoid_backward_axis: float

    # Scaling factor for all the axes of the composed ellipsoid
    ellipsoid_scaling: float

    @staticmethod
    def build(feet_frames: Dict,
              ellipsoid_forward_axis: float = 1.0,
              ellipsoid_side_axis: float = 0.9,
              ellipsoid_backward_axis: float = 0.6,
              ellipsoid_scaling: float = 0.4) -> "Plotter":
        """Build an instance of Plotter."""

        # Default footsteps color: blue
        footsteps_colors = {feet_frames["left_foot"]: 'b', feet_frames["right_foot"]: 'b'}

        return Plotter(feet_frames=feet_frames,
                       footsteps_colors=footsteps_colors,
                       ellipsoid_forward_axis=ellipsoid_forward_axis,
                       ellipsoid_side_axis=ellipsoid_side_axis,
                       ellipsoid_backward_axis=ellipsoid_backward_axis,
                       ellipsoid_scaling=ellipsoid_scaling)

    @staticmethod
    def plot_blending_coefficients(figure_blending_coefficients: int, blending_coeffs: Dict) -> None:
        """Plot the activations of the blending coefficients."""

        plt.figure(figure_blending_coefficients)
        plt.clf()

        # Plot blending coefficients
        plt.plot(range(len(blending_coeffs["w_1"])), blending_coeffs["w_1"], 'r')
        plt.plot(range(len(blending_coeffs["w_2"])), blending_coeffs["w_2"], 'b')
        plt.plot(range(len(blending_coeffs["w_3"])), blending_coeffs["w_3"], 'g')
        plt.plot(range(len(blending_coeffs["w_4"])), blending_coeffs["w_4"], 'y')

        # Plot configuration
        plt.title("Blending coefficients profiles")
        plt.ylabel("Blending coefficients")
        plt.xlabel("Time [s]")

    def plot_new_footstep(self, figure_footsteps: int, support_foot: str, new_footstep: Dict) -> None:
        """Plot a new footstep just added to the footsteps list."""

        plt.figure(figure_footsteps)

        # Footstep position
        plt.scatter(new_footstep["pos"][1], -new_footstep["pos"][0], c=self.footsteps_colors[support_foot])

        # Footstep orientation (scaled for visualization purposes)
        R = Rotation.from_quat(Quaternion.to_xyzw(np.asarray(new_footstep["quat"])))
        RPY = Rotation.as_euler(R, 'xyz')
        yaw = RPY[2]
        plt.plot([new_footstep["pos"][1], new_footstep["pos"][1] + math.sin(yaw) / 5],
                 [-new_footstep["pos"][0], -new_footstep["pos"][0] - math.cos(yaw) / 5],
                 self.footsteps_colors[support_foot])

        # Plot configuration
        plt.axis('scaled')
        plt.title("Footsteps")

        # Plot
        plt.show()
        plt.pause(0.5)

    @staticmethod
    def plot_predicted_future_trajectory(figure_facing_dirs: int, figure_base_vel: int, denormalized_current_output: List) -> None:
        """Plot the future trajectory predicted by the network (magenta)."""

        # Retrieve predicted base positions, facing directions and base velocities from the denormalized network output
        predicted_base_pos = denormalized_current_output[0:12]
        predicted_facing_dirs = denormalized_current_output[12:24]
        predicted_base_vel = denormalized_current_output[24:36]

        plt.figure(figure_facing_dirs)

        for k in range(0, len(predicted_base_pos), 2):

            # Plot base positions
            base_position = [predicted_base_pos[k], predicted_base_pos[k + 1]]
            plt.scatter(-base_position[1], base_position[0], c='m')

            # Plot facing directions (scaled for visualization purposes)
            facing_direction = [predicted_facing_dirs[k] / 10, predicted_facing_dirs[k + 1] / 10]
            plt.plot([-base_position[1], -base_position[1] - facing_direction[1]],
                     [base_position[0], base_position[0] + facing_direction[0]],
                     'm')

        plt.figure(figure_base_vel)

        for k in range(0, len(predicted_base_pos), 2):

            # Plot base positions
            base_position = [predicted_base_pos[k], predicted_base_pos[k + 1]]
            plt.scatter(-base_position[1], base_position[0], c='m')

            # Plot base velocities (scaled for visualization purposes)
            base_velocity = [predicted_base_vel[k] / 10, predicted_base_vel[k + 1] / 10]
            plt.plot([-base_position[1], -base_position[1] - base_velocity[1]],
                     [base_position[0], base_position[0] + base_velocity[0]],
                     'm')

    @staticmethod
    def plot_desired_future_trajectory(figure_facing_dirs: int, figure_base_vel: int,
                                       quad_bezier: List, facing_dirs: List, base_velocities: List) -> None:
        """Plot the future trajectory built from user inputs (gray)."""

        # Retrieve components for plotting
        quad_bezier_x = [elem[0] for elem in quad_bezier]
        quad_bezier_y = [elem[1] for elem in quad_bezier]

        plt.figure(figure_facing_dirs)

        # Plot base positions
        plt.scatter(quad_bezier_x, quad_bezier_y, c='gray')

        # Plot facing directions (scaled for visualization purposes)
        for k in range(len(quad_bezier)):
            plt.plot([quad_bezier_x[k], quad_bezier_x[k] + facing_dirs[k][0] / 10],
                     [quad_bezier_y[k], quad_bezier_y[k] + facing_dirs[k][1] / 10],
                     c='gray')

        plt.figure(figure_base_vel)

        # Plot base positions
        plt.scatter(quad_bezier_x, quad_bezier_y, c='gray')

        # Plot base velocities (scaled for visualization purposes)
        for k in range(len(quad_bezier)):
            plt.plot([quad_bezier_x[k], quad_bezier_x[k] + base_velocities[k][0] / 10],
                     [quad_bezier_y[k], quad_bezier_y[k] + base_velocities[k][1] / 10],
                     c='gray')

    @staticmethod
    def plot_blended_future_trajectory(figure_facing_dirs: int, figure_base_vel: int, blended_base_positions: List,
                                       blended_facing_dirs: List, blended_base_velocities: List) -> None:
        """Plot the future trajectory obtained by blending the network output and the user input (green)."""

        # Extract components for plotting
        blended_base_positions_x = [elem[0] for elem in blended_base_positions]
        blended_base_positions_y = [elem[1] for elem in blended_base_positions]

        plt.figure(figure_facing_dirs)

        # Plot base positions
        plt.scatter(blended_base_positions_x, blended_base_positions_y, c='g')

        # Plot facing directions (scaled for visualization purposes)
        for k in range(len(blended_base_positions)):
            plt.plot([blended_base_positions_x[k], blended_base_positions_x[k] + blended_facing_dirs[k][0] / 10],
                     [blended_base_positions_y[k], blended_base_positions_y[k] + blended_facing_dirs[k][1] / 10],
                     c='g')

        plt.figure(figure_base_vel)

        # Plot base positions
        plt.scatter(blended_base_positions_x, blended_base_positions_y, c='g')

        # Plot base velocities (scaled for visualization purposes)
        for k in range(len(blended_base_positions)):
            plt.plot([blended_base_positions_x[k], blended_base_positions_x[k] + blended_base_velocities[k][0] / 10],
                     [blended_base_positions_y[k], blended_base_positions_y[k] + blended_base_velocities[k][1] / 10],
                     c='g')

    def plot_trajectory_blending(self, figure_facing_dirs: int, figure_base_vel: int, denormalized_current_output: List,
                                 quad_bezier: List, facing_dirs: List, base_velocities: List, blended_base_positions: List,
                                 blended_facing_dirs: List, blended_base_velocities: List) -> None:
        """Plot the predicted, desired and blended future ground trajectories used to build the next network input."""

        # Facing directions plot
        plt.figure(figure_facing_dirs)
        plt.clf()

        # Plot the reference frame
        plt.scatter(0, 0, c='k')
        plt.plot([0, 0], [0, 1 / 10], 'k')

        # Plot upper semi-ellipse of the composed ellipsoid on which the last point of the Bezier curve is constrained
        a = self.ellipsoid_side_axis * self.ellipsoid_scaling
        b = self.ellipsoid_forward_axis * self.ellipsoid_scaling
        x_coord = np.linspace(-a, a, 1000)
        y_coord = b * np.sqrt( 1 - (x_coord ** 2)/(a ** 2))
        plt.plot(x_coord, y_coord, 'k')

        # Plot lower semi-ellipse of the composed ellipsoid on which the last point of the Bezier curve is constrained
        a = self.ellipsoid_side_axis * self.ellipsoid_scaling
        b = self.ellipsoid_backward_axis * self.ellipsoid_scaling
        x_coord = np.linspace(-a, a, 1000)
        y_coord = b * np.sqrt( 1 - (x_coord ** 2)/(a ** 2))
        plt.plot(x_coord, -y_coord, 'k')

        # Base velocities plot
        plt.figure(figure_base_vel)
        plt.clf()

        # Plot the reference frame
        plt.scatter(0, 0, c='k')
        plt.plot([0, 0], [0, 1 / 10], 'k')

        # Plot upper semi-ellipse of the composed ellipsoid on which the last point of the Bezier curve is constrained
        a = self.ellipsoid_side_axis * self.ellipsoid_scaling
        b = self.ellipsoid_forward_axis * self.ellipsoid_scaling
        x_coord = np.linspace(-a, a, 1000)
        y_coord = b * np.sqrt( 1 - (x_coord ** 2)/(a ** 2))
        plt.plot(x_coord, y_coord, 'k')

        # Plot lower semi-ellipse of the composed ellipsoid on which the last point of the Bezier curve is constrained
        a = self.ellipsoid_side_axis * self.ellipsoid_scaling
        b = self.ellipsoid_backward_axis * self.ellipsoid_scaling
        x_coord = np.linspace(-a, a, 1000)
        y_coord = b * np.sqrt( 1 - (x_coord ** 2)/(a ** 2))
        plt.plot(x_coord, -y_coord, 'k')

        # Plot the future trajectory predicted by the network
        self.plot_predicted_future_trajectory(figure_facing_dirs=figure_facing_dirs, figure_base_vel=figure_base_vel,
                                              denormalized_current_output=denormalized_current_output)

        # Plot the future trajectory built from user inputs
        self.plot_desired_future_trajectory(figure_facing_dirs=figure_facing_dirs, figure_base_vel=figure_base_vel,
                                            quad_bezier=quad_bezier, facing_dirs=facing_dirs, base_velocities=base_velocities)

        # Plot the future trajectory obtained by blending the network output and the user input
        self.plot_blended_future_trajectory(figure_facing_dirs=figure_facing_dirs, figure_base_vel=figure_base_vel,
                                            blended_base_positions=blended_base_positions,
                                            blended_facing_dirs=blended_facing_dirs,
                                            blended_base_velocities=blended_base_velocities)

        # Configure facing directions plot
        plt.figure(figure_facing_dirs)
        plt.axis('scaled')
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.3, 0.5])
        plt.axis('off')
        plt.title("INTERPOLATED TRAJECTORY (FACING DIRECTIONS)")

        # Configure base velocities plot
        plt.figure(figure_base_vel)
        plt.axis('scaled')
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.3, 0.5])
        plt.axis('off')
        plt.title("INTERPOLATED TRAJECTORY (BASE VELOCITIES)")

    def reset_from_mergepoint(self):
        """Reset the color to print the footsteps when the generation restarts from the last mergepoint."""

        # Define a random color for the footsteps that is different from the current one
        current_color = self.footsteps_colors[self.feet_frames["left_foot"]]
        colors = ['b','g','r','c','m','y','k']
        colors.remove(current_color)
        random_color = random.choice(colors)
        # print("current color:", current_color, "random color:", random_color,)

        # Reset the colors to print the footsteps
        self.footsteps_colors = {self.feet_frames["left_foot"]: random_color,
                                 self.feet_frames["right_foot"]: random_color}


@dataclass
class LearnedModel:
    """Class for the direct exploitation of the model learned during training."""

    # Path to the learned model
    model_path: str

    # Learned model
    learned_model: None

    # Output mean and standard deviation
    Ymean: List
    Ystd: List

    @staticmethod
    def build(training_path: str) -> "LearnedModel":
        """Build an instance of LearnedModel."""

        # Retrieve path to the latest saved model
        model_path = get_latest_model_path(training_path+"models/")

        # Restore the model with the trained weights
        learned_model = torch.load(model_path)

        # Set dropout and batch normalization layers to evaluation mode before running inference
        learned_model.eval()

        # Compute output mean and standard deviation
        datapath = os.path.join(training_path, "normalization/")
        Ymean, Ystd = load_output_mean_and_std(datapath)

        return LearnedModel(model_path=model_path, learned_model=learned_model, Ymean=Ymean, Ystd=Ystd)


@dataclass
class Autoregression:
    """Class to use the network output, blended with the user-specified input, in an autoregressive fashion."""

    # Component-wise input mean and standard deviation
    Xmean_dict: Dict
    Xstd_dict: Dict

    # Robot-specific frontal base and chest directions
    frontal_base_direction: List
    frontal_chest_direction: List

    # Predefined norm for the base velocities
    base_vel_norm: float

    # Blending parameters tau
    tau_base_positions: float
    tau_facing_dirs: float
    tau_base_velocities: float

    # Auxiliary variable to handle unnatural in-place rotations when the robot is stopped
    nn_X_difference_norm_threshold: float

    # Variables to store autoregression-relevant information for the current iteration
    current_nn_X: List
    current_past_trajectory_base_positions: List
    current_past_trajectory_facing_directions: List
    current_past_trajectory_base_velocities: List
    current_base_position: np.array
    current_base_yaw: float
    current_ground_base_position: List = field(default_factory=lambda: [0,0])
    current_facing_direction: List = field(default_factory=lambda: [1,0])
    current_world_R_facing: np.array = field(default_factory=lambda: np.array([[1, 0], [0, 1]]))

    # Variables to store autoregression-relevant information for the next iteration
    next_nn_X: List = field(default_factory=list)
    new_past_trajectory_base_positions: List = field(default_factory=list)
    new_past_trajectory_facing_directions: List = field(default_factory=list)
    new_past_trajectory_base_velocities: List = field(default_factory=list)
    new_base_position: List = field(default_factory=list)
    new_facing_direction: List = field(default_factory=list)
    new_world_R_facing: List = field(default_factory=list)
    new_facing_R_world: List = field(default_factory=list)
    new_ground_base_position: List = field(default_factory=list)
    new_base_yaw: List = field(default_factory=list)

    # Variables to store autoregression-relevant information at the last mergepoint
    mergepoint_nn_X: List = field(default_factory=list)
    mergepoint_past_trajectory_base_positions: List = field(default_factory=list)
    mergepoint_past_trajectory_facing_directions: List = field(default_factory=list)
    mergepoint_past_trajectory_base_velocities: List = field(default_factory=list)
    mergepoint_base_position: List = field(default_factory=list)
    mergepoint_facing_direction: List = field(default_factory=list)
    mergepoint_world_R_facing: List = field(default_factory=list)
    mergepoint_facing_R_world: List = field(default_factory=list)
    mergepoint_ground_base_position: List = field(default_factory=list)
    mergepoint_base_yaw: List = field(default_factory=list)

    # Number of points constituting the Bezier curve
    t: List = field(default_factory=lambda: np.linspace(0, 1, 7))

    # Relevant indexes of the window storing past data
    past_window_indexes: List = field(default_factory=lambda: [0, 10, 20, 30, 40, 50])

    # Auxiliary variable for the robot status (moving or stopped)
    stopped: bool = True

    @staticmethod
    def build(training_path: str,
              initial_nn_X: List,
              initial_past_trajectory_base_pos: List,
              initial_past_trajectory_facing_dirs: List,
              initial_past_trajectory_base_vel: List,
              initial_base_height: List,
              initial_base_yaw: float,
              frontal_base_direction: List,
              frontal_chest_direction: List,
              base_vel_norm: float = 0.4,
              tau_base_positions: float = 1.5,
              tau_facing_dirs: float = 1.3,
              tau_base_velocities: float = 1.3,
              nn_X_difference_norm_threshold: float = 0.05) -> "Autoregression":
        """Build an instance of Autoregression."""

        # Compute component-wise input mean and standard deviation
        datapath = os.path.join(training_path, "normalization/")
        Xmean_dict, Xstd_dict = load_component_wise_input_mean_and_std(datapath)

        return Autoregression(Xmean_dict=Xmean_dict,
                              Xstd_dict=Xstd_dict,
                              frontal_base_direction=frontal_base_direction,
                              frontal_chest_direction=frontal_chest_direction,
                              base_vel_norm=base_vel_norm,
                              tau_base_positions=tau_base_positions,
                              tau_facing_dirs=tau_facing_dirs,
                              tau_base_velocities=tau_base_velocities,
                              nn_X_difference_norm_threshold=nn_X_difference_norm_threshold,
                              current_nn_X=initial_nn_X,
                              current_past_trajectory_base_positions=initial_past_trajectory_base_pos,
                              current_past_trajectory_facing_directions=initial_past_trajectory_facing_dirs,
                              current_past_trajectory_base_velocities=initial_past_trajectory_base_vel,
                              current_base_position=np.array([0, 0, initial_base_height]),
                              current_base_yaw=initial_base_yaw)

    def update_reference_frame(self, world_H_base: np.array, base_H_chest: np.array) -> None:
        """Update the local reference frame given by the new base position and the new facing direction."""

        # Store new base position
        self.new_base_position = world_H_base[0:3, -1]
        self.new_ground_base_position = [self.new_base_position[0], self.new_base_position[1]] # projected on the ground

        # Retrieve new ground base direction
        new_base_rotation = world_H_base[0:3, 0:3]
        new_base_direction = new_base_rotation.dot(self.frontal_base_direction)
        new_ground_base_direction = [new_base_direction[0], new_base_direction[1]]  # projected on the ground
        new_ground_base_direction = new_ground_base_direction / np.linalg.norm(new_ground_base_direction)  # of unitary norm

        # Retrieve new ground chest direction
        world_H_chest = world_H_base.dot(base_H_chest)
        new_chest_rotation = world_H_chest[0:3, 0:3]
        new_chest_direction = new_chest_rotation.dot(self.frontal_chest_direction)
        new_ground_chest_direction = [new_chest_direction[0], new_chest_direction[1]]  # projected on the ground
        new_ground_chest_direction = new_ground_chest_direction / np.linalg.norm(new_ground_chest_direction)  # of unitary norm

        # Store new facing direction
        self.new_facing_direction = new_ground_base_direction + new_ground_chest_direction  # mean of base and chest directions
        self.new_facing_direction = self.new_facing_direction / np.linalg.norm(self.new_facing_direction)  # of unitary norm

        # Retrieve the rotation from the new facing direction to the world frame and its inverse
        new_facing_direction_yaw = compute_angle_wrt_x_positive_semiaxis(self.new_facing_direction)
        self.new_world_R_facing = rotation_2D(new_facing_direction_yaw)
        self.new_facing_R_world = np.linalg.inv(self.new_world_R_facing)

    def autoregressive_usage_base_positions(self, next_nn_X: List, denormalized_current_output: np.array,
                                            quad_bezier: List) -> (List, List, List):
        """Use the base positions in an autoregressive fashion."""

        # ===================
        # PAST BASE POSITIONS
        # ===================

        # Update the full window storing the past base positions
        new_past_trajectory_base_positions = []
        for k in range(len(self.current_past_trajectory_base_positions) - 1):
            # Element in the reference frame defined by the previous base position + facing direction
            facing_elem = self.current_past_trajectory_base_positions[k + 1]
            # Express element in world frame
            world_elem = self.current_world_R_facing.dot(facing_elem) + self.current_ground_base_position
            # Express element in the frame defined by the new base position + facing direction
            new_facing_elem = self.new_facing_R_world.dot(world_elem - self.new_ground_base_position)
            # Store updated element
            new_past_trajectory_base_positions.append(new_facing_elem)

        # Add as last element the current (local) base position, i.e. [0,0]
        new_past_trajectory_base_positions.append(np.array([0., 0.]))

        # Update past base positions
        self.new_past_trajectory_base_positions = new_past_trajectory_base_positions

        # Extract compressed window of past base positions (denormalized for plotting)
        past_base_positions_plot = []
        for index in self.past_window_indexes:
            past_base_positions_plot.extend(self.new_past_trajectory_base_positions[index])

        # Extract compressed window of past base positions (normalized for building the next input)
        past_base_positions = past_base_positions_plot.copy()
        for k in range(len(past_base_positions)):
            past_base_positions[k] = (past_base_positions[k] - self.Xmean_dict["past_base_positions"][k]) / \
                                     self.Xstd_dict["past_base_positions"][k]

        # Add the compressed window of normalized past base positions to the next input
        next_nn_X.extend(past_base_positions)

        # =====================
        # FUTURE BASE POSITIONS
        # =====================

        # Extract future base positions for blending (i.e. in the plot reference frame)
        future_base_pos_plot = denormalized_current_output[0:12]
        future_base_pos_blend = [[0.0, 0.0]]
        for k in range(0, len(future_base_pos_plot), 2):
            future_base_pos_blend.append([-future_base_pos_plot[k + 1], future_base_pos_plot[k]])

        # Blend user-specified and network-predicted future base positions
        blended_base_positions = trajectory_blending(future_base_pos_blend, quad_bezier, self.t, self.tau_base_positions)

        # Reshape blended future base positions
        future_base_pos_blend_features = []
        for k in range(1, len(blended_base_positions)):
            future_base_pos_blend_features.append(blended_base_positions[k][1])
            future_base_pos_blend_features.append(-blended_base_positions[k][0])

        # Normalize blended future base positions
        future_base_pos_blend_features_normalized = future_base_pos_blend_features.copy()
        for k in range(len(future_base_pos_blend_features_normalized)):
            future_base_pos_blend_features_normalized[k] = (future_base_pos_blend_features_normalized[k] -
                                                            self.Xmean_dict["future_base_positions"][k]) / \
                                                           self.Xstd_dict["future_base_positions"][k]

        # Add the normalized blended future base positions to the next input
        next_nn_X.extend(future_base_pos_blend_features_normalized)

        return next_nn_X, blended_base_positions, future_base_pos_blend_features

    def autoregressive_usage_facing_directions(self, next_nn_X: List, denormalized_current_output: np.array,
                                               facing_dirs: List) -> (List, List):
        """Use the facing directions in an autoregressive fashion."""

        # ======================
        # PAST FACING DIRECTIONS
        # ======================

        # Update the full window storing the past facing directions
        new_past_trajectory_facing_directions = []
        for k in range(len(self.current_past_trajectory_facing_directions) - 1):
            # Element in the reference frame defined by the previous base position + facing direction
            facing_elem = self.current_past_trajectory_facing_directions[k + 1]
            # Express element in world frame
            world_elem = self.current_world_R_facing.dot(facing_elem)
            # Express element in the frame defined by the new base position + facing direction
            new_facing_elem = self.new_facing_R_world.dot(world_elem)
            # Store updated element
            new_past_trajectory_facing_directions.append(new_facing_elem)

        # Add as last element the current (local) facing direction, i.e. [1,0]
        new_past_trajectory_facing_directions.append(np.array([1., 0.]))

        # Update past facing directions
        self.new_past_trajectory_facing_directions = new_past_trajectory_facing_directions

        # Extract compressed window of past facing directions (denormalized for plotting)
        past_facing_directions_plot = []
        for index in self.past_window_indexes:
            past_facing_directions_plot.extend(self.new_past_trajectory_facing_directions[index])

        # Extract compressed window of past facing directions (normalized for building the next input)
        past_facing_directions = past_facing_directions_plot.copy()
        for k in range(len(past_facing_directions)):
            past_facing_directions[k] = (past_facing_directions[k] - self.Xmean_dict["past_facing_directions"][k]) / \
                                        self.Xstd_dict["past_facing_directions"][k]

        # Add the compressed window of normalized past facing directions to the next input
        next_nn_X.extend(past_facing_directions)

        # ========================
        # FUTURE FACING DIRECTIONS
        # ========================

        # Extract future facing directions for blending (i.e. in the plot reference frame)
        future_facing_dirs_plot = denormalized_current_output[12:24]
        future_facing_dirs_blend = [[0.0, 1.0]]
        for k in range(0, len(future_facing_dirs_plot), 2):
            future_facing_dirs_blend.append([-future_facing_dirs_plot[k + 1], future_facing_dirs_plot[k]])

        # Blend user-specified and network-predicted future facing directions
        blended_facing_dirs = trajectory_blending(future_facing_dirs_blend, facing_dirs, self.t, self.tau_facing_dirs)

        # Reshape blended future facing directions
        future_facing_dirs_blend_features = []
        for k in range(1, len(blended_facing_dirs)):
            future_facing_dirs_blend_features.append(blended_facing_dirs[k][1])
            future_facing_dirs_blend_features.append(-blended_facing_dirs[k][0])

        # Normalize blended future facing directions
        future_facing_dirs_blend_features_normalized = future_facing_dirs_blend_features.copy()
        for k in range(len(future_facing_dirs_blend_features_normalized)):
            future_facing_dirs_blend_features_normalized[k] = (future_facing_dirs_blend_features_normalized[k] -
                                                               self.Xmean_dict["future_facing_directions"][k]) / \
                                                              self.Xstd_dict["future_facing_directions"][k]

        # Add the normalized blended future facing directions to the next input
        next_nn_X.extend(future_facing_dirs_blend_features_normalized)

        return next_nn_X, blended_facing_dirs

    def autoregressive_usage_base_velocities(self, next_nn_X: List, denormalized_current_output: np.array,
                                             base_velocities: List) -> (List, List):
        """Use the base velocities in an autoregressive fashion."""

        # ====================
        # PAST BASE VELOCITIES
        # ====================

        # Update the full window storing the past base velocities
        new_past_trajectory_base_velocities = []
        for k in range(len(self.current_past_trajectory_base_velocities) - 1):
            # Element in the reference frame defined by the previous base position + facing direction
            facing_elem = self.current_past_trajectory_base_velocities[k + 1]
            # Express element in world frame
            world_elem = self.current_world_R_facing.dot(facing_elem)
            # Express element in the frame defined by the new base position + facing direction
            new_facing_elem = self.new_facing_R_world.dot(world_elem)
            # Store updated element
            new_past_trajectory_base_velocities.append(new_facing_elem)

        # Add as last element the current (local) base velocity (this is an approximation)
        new_past_trajectory_base_velocities.append(self.new_facing_R_world.dot(rotation_2D(self.new_base_yaw).dot(
            [denormalized_current_output[100], denormalized_current_output[101]])))

        # Update past base velocities
        self.new_past_trajectory_base_velocities = new_past_trajectory_base_velocities

        # Extract compressed window of past base velocities (denormalized for plotting)
        past_base_velocities_plot = []
        for index in self.past_window_indexes:
            past_base_velocities_plot.extend(self.new_past_trajectory_base_velocities[index])

        # Extract compressed window of past base velocities (normalized for building the next input)
        past_base_velocities = past_base_velocities_plot.copy()
        for k in range(len(past_base_velocities)):
            past_base_velocities[k] = (past_base_velocities[k] - self.Xmean_dict["past_base_velocities"][k]) / \
                                      self.Xstd_dict["past_base_velocities"][k]

        # Add the compressed window of normalized past ground base velocities to the next input
        next_nn_X.extend(past_base_velocities)

        # ======================
        # FUTURE BASE VELOCITIES
        # ======================

        # Extract future base velocities for blending (i.e. in the plot reference frame)
        future_base_vel_plot = denormalized_current_output[24:36]
        future_base_vel_blend = [[0.0, self.base_vel_norm]] # This is an approximation.
        for k in range(0, len(future_base_vel_plot), 2):
            future_base_vel_blend.append([-future_base_vel_plot[k + 1], future_base_vel_plot[k]])

        # blend user-specified and network-predicted future base velocities
        blended_base_velocities = trajectory_blending(future_base_vel_blend, base_velocities, self.t, self.tau_base_velocities)

        # Reshape blended future base velocities
        future_base_velocities_blend_features = []
        for k in range(1, len(blended_base_velocities)):
            future_base_velocities_blend_features.append(blended_base_velocities[k][1])
            future_base_velocities_blend_features.append(-blended_base_velocities[k][0])

        # Normalize future base velocities
        future_base_velocities_blend_features_normalized = future_base_velocities_blend_features.copy()
        for k in range(len(future_base_velocities_blend_features_normalized)):
            future_base_velocities_blend_features_normalized[k] = (future_base_velocities_blend_features_normalized[k] -
                                                                   self.Xmean_dict["future_base_velocities"][k]) / \
                                                                  self.Xstd_dict["future_base_velocities"][k]

        # Add the normalized blended future base velocities to the next input
        next_nn_X.extend(future_base_velocities_blend_features_normalized)

        return next_nn_X, blended_base_velocities

    def autoregressive_usage_future_traj_len(self, next_nn_X: List, future_base_pos_blend_features: List) -> List:
        """Use the future length trajectory in an autoregressive fashion."""

        # Compute the desired future trajectory length by summing the distances between future base positions
        future_traj_length = 0
        future_base_position_prev = future_base_pos_blend_features[0]
        for future_base_position in future_base_pos_blend_features[1:]:
            base_position_distance = np.linalg.norm(future_base_position - future_base_position_prev)
            future_traj_length += base_position_distance
            future_base_position_prev = future_base_position

        # Normalize the desired future trajectory length
        future_traj_length = future_traj_length - self.Xmean_dict["future_traj_length"] / self.Xstd_dict["future_traj_length"]

        # Add the desired future trajectory length to the next input
        next_nn_X.extend([future_traj_length])

        return next_nn_X

    def autoregressive_usage_joint_positions_and_velocities(self, next_nn_X: List, current_output: np.array) -> List:
        """Use the joint positions and velocities in an autoregressive fashion."""

        # Add the (already normalized) joint positions to the next input
        s = current_output[0][36:68]
        next_nn_X.extend(s)

        # Add the (already normalized) joint velocities to the next input
        s_dot = current_output[0][68:100]
        next_nn_X.extend(s_dot)

        return next_nn_X

    def check_robot_stopped(self, next_nn_X: List) -> None:
        """Check whether the robot is stopped (i.e. whether subsequent network inputs are almost identical)."""

        # Compute the difference in norm between the current and the next network inputs
        nn_X_difference_norm = np.linalg.norm(np.array(self.current_nn_X[0]) - np.array(next_nn_X))

        # The robot is considered to be stopped if the difference in norm is lower than a threshold
        if nn_X_difference_norm < self.nn_X_difference_norm_threshold:
            self.stopped = True
        else:
            self.stopped = False

    def update_autoregression_state(self, next_nn_X: List) -> None:
        """Update the autoregression-relevant information."""

        self.current_nn_X = [next_nn_X]
        self.current_past_trajectory_base_positions = self.new_past_trajectory_base_positions
        self.current_past_trajectory_facing_directions = self.new_past_trajectory_facing_directions
        self.current_past_trajectory_base_velocities = self.new_past_trajectory_base_velocities
        self.current_facing_direction = self.new_facing_direction
        self.current_world_R_facing = self.new_world_R_facing
        self.current_base_position = self.new_base_position
        self.current_ground_base_position = self.new_ground_base_position
        self.current_base_yaw = self.new_base_yaw

    def autoregression_and_blending(self, current_output: np.array, denormalized_current_output: np.array,
                                    quad_bezier: List, facing_dirs: List, base_velocities: List,
                                    world_H_base: np.array, base_H_chest: np.array) -> (List, List, List):
        """Handle the autoregressive usage of the network output blended with the user input from the joystick."""

        # Update the bi-dimensional reference frame given by the base position and the facing direction
        self.update_reference_frame(world_H_base=world_H_base, base_H_chest=base_H_chest)

        # Initialize empty next input
        next_nn_X = []

        # Use the base positions in an autoregressive fashion
        next_nn_X, blended_base_positions, future_base_pos_blend_features = \
            self.autoregressive_usage_base_positions(next_nn_X=next_nn_X,
                                                     denormalized_current_output=denormalized_current_output,
                                                     quad_bezier=quad_bezier)

        # Use the facing directions in an autoregressive fashion
        next_nn_X, blended_facing_dirs = \
            self.autoregressive_usage_facing_directions(next_nn_X=next_nn_X,
                                                        denormalized_current_output=denormalized_current_output,
                                                        facing_dirs=facing_dirs)

        # Use the base velocities in an autoregressive fashion
        next_nn_X, blended_base_velocities = \
            self.autoregressive_usage_base_velocities(next_nn_X=next_nn_X,
                                                      denormalized_current_output=denormalized_current_output,
                                                      base_velocities=base_velocities)

        # Use the future trajectory length in an autoregressive fashion
        next_nn_X = self.autoregressive_usage_future_traj_len(next_nn_X=next_nn_X,
                                                              future_base_pos_blend_features=future_base_pos_blend_features)

        # Use the joint positions and velocities in an autoregressive fashion
        next_nn_X = self.autoregressive_usage_joint_positions_and_velocities(next_nn_X, current_output)

        # Check whether the robot is stopped
        self.check_robot_stopped(next_nn_X)

        # Update autoregressive-relevant information for the next iteration
        self.update_autoregression_state(next_nn_X)

        return blended_base_positions, blended_facing_dirs, blended_base_velocities

    def update_mergepoint_state(self) -> None:
        """Update the autoregression-relevant information at the mergepoint."""

        self.mergepoint_nn_X = self.current_nn_X
        self.mergepoint_past_trajectory_base_positions = self.current_past_trajectory_base_positions
        self.mergepoint_past_trajectory_facing_directions = self.current_past_trajectory_facing_directions
        self.mergepoint_past_trajectory_base_velocities = self.current_past_trajectory_base_velocities
        self.mergepoint_facing_direction = self.current_facing_direction
        self.mergepoint_world_R_facing = self.current_world_R_facing
        self.mergepoint_base_position = self.current_base_position
        self.mergepoint_ground_base_position = self.current_ground_base_position
        self.mergepoint_base_yaw = self.current_base_yaw

    def reset_from_mergepoint(self) -> None:
        """Restore the autoregression-relevant information from the last mergepoint."""

        self.current_nn_X = self.mergepoint_nn_X
        self.current_past_trajectory_base_positions = self.mergepoint_past_trajectory_base_positions
        self.current_past_trajectory_facing_directions = self.mergepoint_past_trajectory_facing_directions
        self.current_past_trajectory_base_velocities = self.mergepoint_past_trajectory_base_velocities
        self.current_facing_direction = self.mergepoint_facing_direction
        self.current_world_R_facing = self.mergepoint_world_R_facing
        self.current_base_position = self.mergepoint_base_position
        self.current_ground_base_position = self.mergepoint_ground_base_position
        self.current_base_yaw = self.mergepoint_base_yaw


@dataclass
class TrajectoryGenerator:
    """Class for generating trajectories."""

    # Subcomponents of the trajectory generator
    kincomputations: KinematicComputations
    storage: StorageHandler
    autoregression: Autoregression
    plotter: Plotter
    model: LearnedModel

    # Iteration counter and generation rate
    generation_rate: float
    iteration: int = 0

    # Store whether and when the mergepoint has been reached
    mergepoint_reached: bool = False
    mergepoint_iteration: int = 0
    # Store the number of steps of the current portion of generated trajectory
    n_steps: int = 0
    # Store the number of iterations of the current portion of generated trajectory
    n_iterations: int = 0

    @staticmethod
    def build(icub: iCub,
              gazebo: scenario.GazeboSimulator,
              kindyn: kindyncomputations.KinDynComputations,
              controlled_joints_indexes: List,
              storage_path: str,
              training_path: str,
              local_foot_vertices_pos: List,
              feet_frames: Dict,
              feet_links: Dict,
              initial_nn_X: List,
              initial_past_trajectory_base_pos: List,
              initial_past_trajectory_facing_dirs: List,
              initial_past_trajectory_base_vel: List,
              initial_base_height: List,
              initial_base_yaw: float,
              frontal_base_direction: List,
              frontal_chest_direction: List,
              initial_support_foot: str,
              initial_support_vertex: int,
              initial_l_foot_position: List,
              initial_r_foot_position: List,
              time_scaling: int,
              nominal_DS_duration: float = 0.04,
              difference_position_threshold: float = 0.04,
              difference_height_norm_threshold: bool = 0.005,
              base_vel_norm: float = 0.4,
              tau_base_positions: float = 1.5,
              tau_facing_dirs: float = 1.3,
              tau_base_velocities: float = 1.3,
              nn_X_difference_norm_threshold: float = 0.05,
              ellipsoid_forward_axis: float = 1.0,
              ellipsoid_side_axis: float = 0.9,
              ellipsoid_backward_axis: float = 0.6,
              ellipsoid_scaling: float = 0.4,
              generation_rate: float = 1/50,
              control_rate: float = 1/100) -> "TrajectoryGenerator":
        """Build an instance of TrajectoryGenerator."""

        # Build the kinematic computations handler component
        kincomputations = KinematicComputations.build(kindyn=kindyn,
                                                      controlled_joints_indexes=controlled_joints_indexes,
                                                      local_foot_vertices_pos=local_foot_vertices_pos,
                                                      feet_frames=feet_frames,
                                                      feet_links=feet_links,
                                                      icub=icub,
                                                      gazebo=gazebo,
                                                      initial_support_foot=initial_support_foot,
                                                      initial_support_vertex=initial_support_vertex,
                                                      nominal_DS_duration=nominal_DS_duration,
                                                      difference_position_threshold=difference_position_threshold,
                                                      difference_height_norm_threshold=difference_height_norm_threshold,
                                                      time_scaling=time_scaling)

        # Initialize the support vertex and the support foot
        kincomputations.set_initial_support_vertex_and_support_foot()

        # Build the storage handler component
        generation_to_control_time_scaling = int(generation_rate / control_rate)
        storage = StorageHandler.build(storage_path=storage_path,
                                       feet_frames=feet_frames,
                                       initial_l_foot_position=initial_l_foot_position,
                                       initial_r_foot_position=initial_r_foot_position,
                                       time_scaling=time_scaling,
                                       generation_to_control_time_scaling=generation_to_control_time_scaling)

        # Build the autoregression handler component
        autoregression = Autoregression.build(training_path=training_path,
                                              initial_nn_X=initial_nn_X,
                                              initial_past_trajectory_base_pos=initial_past_trajectory_base_pos,
                                              initial_past_trajectory_facing_dirs=initial_past_trajectory_facing_dirs,
                                              initial_past_trajectory_base_vel=initial_past_trajectory_base_vel,
                                              initial_base_height=initial_base_height,
                                              initial_base_yaw=initial_base_yaw,
                                              frontal_base_direction=frontal_base_direction,
                                              frontal_chest_direction=frontal_chest_direction,
                                              base_vel_norm=base_vel_norm,
                                              tau_base_positions=tau_base_positions,
                                              tau_facing_dirs=tau_facing_dirs,
                                              tau_base_velocities=tau_base_velocities,
                                              nn_X_difference_norm_threshold=nn_X_difference_norm_threshold)

        # Build the plotter component
        plotter = Plotter.build(feet_frames=feet_frames,
                                ellipsoid_forward_axis=ellipsoid_forward_axis,
                                ellipsoid_side_axis=ellipsoid_side_axis,
                                ellipsoid_backward_axis=ellipsoid_backward_axis,
                                ellipsoid_scaling=ellipsoid_scaling)

        # Build the learned model component
        model = LearnedModel.build(training_path=training_path)

        return TrajectoryGenerator(kincomputations=kincomputations,
                                   storage=storage,
                                   autoregression=autoregression,
                                   plotter=plotter,
                                   model=model,
                                   generation_rate=generation_rate)

    def retrieve_network_output_pytorch(self) -> (np.array, np.array):
        """Retrieve the network output (also denormalized)."""

        # Retrieve the network output
        current_output = self.model.learned_model.inference(torch.tensor(self.autoregression.current_nn_X)).numpy()

        # Denormalize the network output
        denormalized_current_output = denormalize(current_output, self.model.Ymean, self.model.Ystd)[0]

        return current_output, denormalized_current_output

    def apply_joint_positions_and_base_orientation(self, denormalized_current_output: List, base_pitch_offset: float = 0) -> (List, List):
        """Apply joint positions and base orientation from the output returned by the network."""

        # Extract the new joint positions from the denormalized network output
        joint_positions = np.asarray(denormalized_current_output[36:68])

        # Extract the joint velocities from the denormalized network output
        joint_velocities = np.asarray(denormalized_current_output[68:100])

        # If the robot is stopped, handle unnatural in-place rotations by imposing zero angular base velocity
        if self.autoregression.stopped:
            omega = 0
        else:
            omega = denormalized_current_output[102]

        # Extract the new base orientation from the output
        base_yaw_dot = omega * self.generation_rate
        new_base_yaw = self.autoregression.current_base_yaw + base_yaw_dot
        new_base_rotation = Rotation.from_euler('xyz', [0, base_pitch_offset, new_base_yaw])
        new_base_quaternion = Quaternion.to_wxyz(new_base_rotation.as_quat())

        # Update the base orientation and the joint positions in the robot configuration
        self.kincomputations.reset_robot_configuration(joint_positions=joint_positions,
                                                       joint_velocities=joint_velocities,
                                                       base_position=self.autoregression.current_base_position,
                                                       base_quaternion=new_base_quaternion)

        # Update the base base orientation and the joint positions in the configuration of the robot visualized in the simulator
        self.kincomputations.reset_visual_robot_configuration(joint_positions=joint_positions,
                                                              base_quaternion=new_base_quaternion)

        # Update the base yaw in the autoregression state
        self.autoregression.new_base_yaw = new_base_yaw

        return joint_positions, joint_velocities, new_base_quaternion

    def update_support_vertex_position(self) -> None:
        """Update the support vertex position."""

        self.kincomputations.update_support_vertex_pos()

    def update_support_vertex_and_support_foot_and_footsteps(self) -> (str, bool):
        """Update the support vertex and the support foot. Handle updates of the footsteps list and of the deactivation
        time of the last footstep."""

        # Update support foot and support vertex while detecting new footsteps and deactivation time updates
        support_foot, update_deactivation_time, update_footsteps_list = self.kincomputations.update_support_vertex_and_support_foot(self.iteration)

        if update_deactivation_time:

            # Define the swing foot
            if support_foot == self.kincomputations.feet_frames["right_foot"]:
                swing_foot = self.kincomputations.feet_frames["left_foot"]
            else:
                swing_foot = self.kincomputations.feet_frames["right_foot"]

            if self.storage.footsteps[swing_foot]:

                # Update the deactivation time of the last footstep
                self.storage.footsteps[swing_foot][-1]["deactivation_time"] = self.iteration * self.generation_rate * self.storage.time_scaling

        if update_footsteps_list:

            # Retrieve the information related to the new footstep
            new_footstep = self.kincomputations.footsteps_extractor.create_new_footstep(
                kindyn=self.kincomputations.kindyn,
                support_foot=support_foot,
                activation_time=self.iteration * self.generation_rate * self.storage.time_scaling)

            # Check close footsteps
            close_footsteps = self.kincomputations.footsteps_extractor.check_close_footsteps(
                current_footstep=self.storage.footsteps[support_foot][-1],
                next_footstep=new_footstep,
            )

            # Debug TODO: remove
            # print("New footstep:", new_footstep)
            # print("Close footsteps?", close_footsteps)
            # input()

            if not close_footsteps:

                # Update the footsteps storage
                self.storage.update_footsteps_storage(support_foot=support_foot, footstep=new_footstep)

                # Debug TODO: remove
                # for foot in self.storage.footsteps.keys():
                #     print("##################################################", foot)
                #     for elem in self.storage.footsteps[foot]:
                #         print(elem)

                # Increase the number of footsteps
                self.n_steps += 1

                # Set the mergepoint after the first footstep
                if self.n_steps == 1:
                    self.mergepoint_reached = True
                    self.mergepoint_iteration = self.iteration
                    self.autoregression.update_mergepoint_state()
                    self.kincomputations.update_mergepoint_state()
                    self.storage.update_mergepoint_state()

            else:

                update_footsteps_list = False

        return support_foot, update_footsteps_list

    def compute_kinematically_fasible_base_and_update_posturals(self, joint_positions: List, joint_velocities: List,
                                                                base_quaternion: List, controlled_joints: List,
                                                                link_names: List) -> (List, List, List, List):
        """Compute kinematically-feasible base position and retrieve updated posturals."""

        # Compute and apply kinematically-feasible base position
        kinematically_feasible_base_position = \
            self.kincomputations.compute_and_apply_kinematically_feasible_base_position( joint_positions=joint_positions,
                                                                                         joint_velocities=joint_velocities,
                                                                                         base_quaternion=base_quaternion)

        # Retrieve new posturals to be added to the list of posturals
        new_base_postural, new_joints_pos_postural, new_joints_vel_postural, new_links_postural, \
        new_com_pos_postural, new_com_vel_postural, new_centroidal_momentum_postural = \
            self.kincomputations.postural_extractor.create_new_posturals(base_position=kinematically_feasible_base_position,
                                                                         base_quaternion=base_quaternion,
                                                                         joint_positions=joint_positions,
                                                                         controlled_joints=controlled_joints,
                                                                         kindyn=self.kincomputations.kindyn,
                                                                         link_names=link_names)

        return new_base_postural, new_joints_pos_postural, new_joints_vel_postural, new_links_postural, \
               new_com_pos_postural, new_com_vel_postural, new_centroidal_momentum_postural

    def retrieve_joystick_inputs(self, input_port: yarp.BufferedPortBottle, quad_bezier: List, base_velocities: List,
                                 facing_dirs: List, raw_data: List) -> (List, List, List, List):
        """Retrieve user-specified joystick inputs received through YARP port."""

        # The joystick input from the user written on the YARP port will contain 3 * 7 * 2 + 4 = 46 values:
        # 0-13 are quad_bezier (x,y)
        # 14-27 are base_velocities (x,y)
        # 28-41 are facing_dirs (x,y)
        # 42-45 are joystick inputs to be stored for future plotting (curr_x, curr_y, curr_z, curr_rz)

        # Read from the input port
        res = input_port.read(shouldWait=False)

        if res is None:

            if quad_bezier:

                # If the port is empty but the previous joystick inputs are not empty, return them
                return quad_bezier, base_velocities, facing_dirs, raw_data

            else:

                # If the port is empty and the previous joystick inputs are empty, return default values
                default_quad_bezier = [[0, 0] for _ in range(len(self.autoregression.t))]
                default_base_velocities = [[0, 0] for _ in range(len(self.autoregression.t))]
                default_facing_dirs = [[0, 1] for _ in range(len(self.autoregression.t))]
                default_raw_data = [0, 0, 0, -1] # zero motion direction (robot stopped), forward facing direction

                return default_quad_bezier, default_base_velocities, default_facing_dirs, default_raw_data

        else:

            # If the port is not empty, retrieve the new joystick inputs
            new_quad_bezier = []
            new_base_velocities = []
            new_facing_dirs = []
            new_raw_data = []

            for k in range(0, res.size() - 4, 2):
                coords = [res.get(k).asFloat32(), res.get(k + 1).asFloat32()]
                if k < 14:
                    new_quad_bezier.append(coords)
                elif k < 28:
                    new_base_velocities.append(coords)
                else:
                    new_facing_dirs.append(coords)

            for k in range(res.size() - 4, res.size()):
                new_raw_data.append(res.get(k).asFloat32())

            return new_quad_bezier, new_base_velocities, new_facing_dirs, new_raw_data

    def autoregression_and_blending(self, current_output: np.array, denormalized_current_output: np.array, quad_bezier: List,
                  facing_dirs: List, base_velocities: List) -> (List, List, List):
        """Use the network output in an autoregressive fashion and blend it with the user input."""

        world_H_base = self.kincomputations.kindyn.get_world_base_transform()
        base_H_chest = self.kincomputations.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="chest")

        # Use the network output in an autoregressive fashion and blend it with the user input
        blended_base_positions, blended_facing_dirs, blended_base_velocities = \
            self.autoregression.autoregression_and_blending(current_output=current_output,
                                                            denormalized_current_output=denormalized_current_output,
                                                            quad_bezier=quad_bezier,
                                                            facing_dirs=facing_dirs,
                                                            base_velocities=base_velocities,
                                                            world_H_base=world_H_base,
                                                            base_H_chest=base_H_chest)

        return blended_base_positions, blended_facing_dirs, blended_base_velocities

    # TODO: remove the save
    def update_storages_and_save(self, blending_coefficients: List, base_postural: List, joints_pos_postural: List,
                                 joint_vel_postural: List, links_postural: List, com_pos_postural: List,
                                 com_vel_postural: List, centroidal_momentum_postural: List, raw_data: List,
                                 quad_bezier: List, base_velocities: List, facing_dirs: List, stream_every_N_iterations: int,
                                 plot_contacts: bool) -> None:
        """Update the blending coefficients, posturals and joystick input storages and periodically save data."""

        # TODO: remove (temporary fix to ignore the storage of the blending coefficients)
        # # Update the blending coefficients storage
        # self.storage.update_blending_coefficients_storage(blending_coefficients=blending_coefficients)

        # Update the posturals storage
        self.storage.update_posturals_storage(base=base_postural, joints_pos=joints_pos_postural,
                                              joints_vel=joint_vel_postural, links=links_postural,
                                              com_pos=com_pos_postural, com_vel=com_vel_postural,
                                              centroidal_momentum=centroidal_momentum_postural)

        # Update joystick inputs storage
        self.storage.update_joystick_inputs_storage(raw_data=raw_data, quad_bezier=quad_bezier,
                                                    base_velocities=base_velocities, facing_dirs=facing_dirs)

        # TODO: remove the save
        # Periodically stream data
        if self.iteration % stream_every_N_iterations == 0:

            # TODO: remove (and do it online if needed)
            # Before saving data, update the footsteps list
            final_deactivation_time = self.iteration * self.generation_rate
            updated_footsteps = self.kincomputations.footsteps_extractor.update_footsteps(
                final_deactivation_time=final_deactivation_time, footsteps=self.storage.footsteps)
            self.storage.replace_footsteps_storage(footsteps=updated_footsteps)

            # TODO: integrate once we define the message to be passed via YARP
            # Retrieve and optionally plot contacts for the controller
            # contact_phase_list = self.storage.retrieve_contacts(plot_contacts=plot_contacts)

            # TODO: remove the save
            # Save data
            # self.storage.save_data_as_json()

    def update_iteration_counter(self) -> None:
        """Update the counter for the iterations of the generator."""

        # Debug
        print(self.iteration)
        if self.iteration == 1:
            input("\nPress Enter to start the trajectory generation.\n")

        self.iteration += 1
        self.n_iterations += 1

    # TODO: also handle missing deactivation times at the beginning ?
    def preprocess_data(self) -> None:
        """Fix last contact deactivation time before streaming data."""

        # Fix deactivation times
        for foot in self.storage.footsteps.keys():
            self.storage.footsteps[foot][-1]["deactivation_time"] = self.iteration * self.generation_rate

    # TODO: stream data (now this is just printing)
    def stream_data(self) -> None:
        """Stream data via a YARP port."""

        print("\n\nDATA STREAM:")

        # Footsteps
        for foot in self.storage.footsteps.keys():
            print("##################################################", foot)
            for elem in self.storage.footsteps[foot]:
                print(elem)

        # Postural
        print("################################################## postural")
        for key in self.storage.posturals.keys():
            print("len (", key , "): " , len(self.storage.posturals[key]))

    def reset_from_mergepoint(self) -> None:
        """Restore the trajectory generator state to the last mergepoint."""

        self.autoregression.reset_from_mergepoint()
        self.kincomputations.reset_from_mergepoint()
        self.plotter.reset_from_mergepoint()
        self.storage.reset_from_mergepoint(updated_activation_time = self.mergepoint_iteration * self.generation_rate)
        self.reset_mergepoint()

    def reset_mergepoint(self) -> None:
        """Reset the state of the trajectory generator at the last mergepoint."""

        self.mergepoint_reached = False
        self.n_steps = 0
        self.n_iterations = 0
        self.iteration = self.mergepoint_iteration

    def get_n_steps(self) -> int:
        """Get the number of steps performed in the last sequence of generated trajectory."""

        return self.n_steps

    def get_n_iterations(self) -> int:
        """Get the number of iterations occurred in the last sequence of generated trajectory."""

        return self.n_iterations
# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import time
import math
import json
import numpy as np
from scenario import core
from typing import List, Dict
from dataclasses import dataclass
from scenario import gazebo as scenario

# =============
# GENERAL UTILS
# =============

def read_from_file(filename: str) -> np.array:
    """Read data as json from file."""

    with open(filename, 'r') as openfile:
        data = json.load(openfile)

    return np.array(data)

def rotation_2D(angle: float) -> np.array:
    """Auxiliary function for a 2-dimensional rotation matrix."""

    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])

# =====================
# MODEL INSERTION UTILS
# =====================

class iCub(core.Model):
    """Helper class to simplify model insertion."""

    def __init__(self,
                 world: scenario.World,
                 urdf: str,
                 position: List[float] = (0., 0, 0),
                 orientation: List[float] = (1., 0, 0, 0)):

        # Insert the model in the world
        name = "iCub"
        pose = core.Pose(position, orientation)
        world.insert_model(urdf, pose, name)

        # Get and store the model from the world
        self.model = world.get_model(model_name=name)

    def __getattr__(self, name):
        return getattr(self.model, name)

# =====================
# JOYSTICK DEVICE UTILS
# =====================

def quadratic_bezier(p0: np.array, p1: np.array, p2: np.array, t: np.array) -> List:
    """Define a discrete quadratic Bezier curve. Given the initial point p0, the control point p1 and
       the final point p2, the quadratic Bezier consists of t points and is defined by:
               Bezier(p0, p1, p2, t) = (1 - t)^2 p0 + 2t (1 - t) p1 + t^2 p2
    """

    quadratic_bezier = []

    for t_i in t:
        p_i = (1 - t_i) * (1 - t_i) * p0 + 2 * t_i * (1 - t_i) * p1 + t_i * t_i * p2
        quadratic_bezier.append(p_i)

    return quadratic_bezier

def compute_angle_wrt_x_positive_semiaxis(current_facing_direction: List) -> float:
    """Compute the angle between the current facing direction and the x positive semiaxis."""

    # Define the x positive semiaxis
    x_positive_semiaxis = np.asarray([1, 0])

    # Compute the yaw between the current facing direction and the world x axis
    cos_theta = np.dot(x_positive_semiaxis, current_facing_direction) # unitary norm vectors
    sin_theta = np.cross(x_positive_semiaxis, current_facing_direction) # unitary norm vectors
    angle = math.atan2(sin_theta, cos_theta)

    return angle

# ===========================
# TRAJECTORY GENERATION UTILS
# ===========================

def trajectory_blending(a0: List, a1: List, t: np.array, tau: float) -> List:
    """Blend the vectors a0 and a1 via:
           Blend(a0, a1, t, tau) = (1 - t^tau) a0 + t^tau a1
       Increasing tau means biasing more towards a1.
    """

    blended_trajectory = []

    for i in range(len(t)):
        p_i = (1 - math.pow(t[i], tau)) * np.array(a0[i]) + math.pow(t[i], tau) * np.array(a1[i])
        blended_trajectory.append(p_i.tolist())

    return blended_trajectory

def load_component_wise_input_mean_and_std(datapath: str) -> (Dict, Dict):
    """Compute component-wise input mean and standard deviation."""

    # Full-input mean and std
    Xmean = read_from_file(datapath + 'X_mean.txt')
    Xstd = read_from_file(datapath + 'X_std.txt')

    # Remove zeroes from Xstd
    for i in range(Xstd.size):
        if Xstd[i] == 0:
            Xstd[i] = 1

    # Retrieve component-wise input mean and std (used to normalize the next input for the network)
    Xmean_dict = {"past_base_positions": Xmean[0:12]}
    Xstd_dict = {"past_base_positions": Xstd[0:12]}
    Xmean_dict["future_base_positions"] = Xmean[12:24]
    Xstd_dict["future_base_positions"] = Xstd[12:24]
    Xmean_dict["past_facing_directions"] = Xmean[24:36]
    Xstd_dict["past_facing_directions"] = Xstd[24:36]
    Xmean_dict["future_facing_directions"] = Xmean[36:48]
    Xstd_dict["future_facing_directions"] = Xstd[36:48]
    Xmean_dict["past_base_velocities"] = Xmean[48:60]
    Xstd_dict["past_base_velocities"] = Xstd[48:60]
    Xmean_dict["future_base_velocities"] = Xmean[60:72]
    Xstd_dict["future_base_velocities"] = Xstd[60:72]
    # Xmean_dict["future_traj_length"] = Xmean[72]
    # Xstd_dict["future_traj_length"] = Xstd[72]
    Xmean_dict["s"] = Xmean[72:98]
    Xstd_dict["s"] = Xstd[72:98]
    Xmean_dict["s_dot"] = Xmean[98:]
    Xstd_dict["s_dot"] = Xstd[98:]

    return Xmean_dict, Xstd_dict

def load_output_mean_and_std(datapath: str) -> (List, List):
    """Compute output mean and standard deviation."""

    # Full-output mean and std
    Ymean = read_from_file(datapath + 'Y_mean.txt')
    Ystd = read_from_file(datapath + 'Y_std.txt')

    # Remove zeroes from Ystd
    for i in range(Ystd.size):
        if Ystd[i] == 0:
            Ystd[i] = 1

    return Ymean, Ystd

def load_input_mean_and_std(datapath: str) -> (List, List):
    """Compute input mean and standard deviation."""

    # Full-input mean and std
    Xmean = read_from_file(datapath + 'X_mean.txt')
    Xstd = read_from_file(datapath + 'X_std.txt')

    # Remove zeroes from Xstd
    for i in range(Xstd.size):
        if Xstd[i] == 0:
            Xstd[i] = 1

    return Xmean, Xstd

def define_base_pitch_offset(robot: str) -> List:
    """Define the robot-specific pitch offset for the base frame."""

    if robot == "ergoCubV1":
        base_pitch_offset = - 0.08

    else:
        raise Exception("Base pitch offset only defined for ergoCubV1.")

    return base_pitch_offset

def define_feet_frames_and_links(robot: str) -> Dict:
    """Define the robot-specific feet frames and links."""

    if robot == "ergoCubV1":
        right_foot_frame = "r_sole"
        left_foot_frame = "l_sole"
        right_foot_link = "r_ankle_2"
        left_foot_link = "l_ankle_2"

    else:
        raise Exception("Feet frames and links only defined ergoCubV1.")

    feet_frames = {"right_foot": right_foot_frame, "left_foot": left_foot_frame}
    feet_links = {feet_frames["right_foot"]: right_foot_link, feet_frames["left_foot"]: left_foot_link}

    return feet_frames, feet_links

def define_foot_vertices(robot: str) -> List:
    """Define the robot-specific positions of the feet vertices in the foot frame."""

    if robot == "ergoCubV1":

        # For ergoCubV1, the feet vertices are symmetrically placed wrt the sole frame origin.
        # The sole frame has z pointing up, x pointing forward and y pointing left.

        # Size of the box which represents the foot rear
        box_size = [0.117, 0.1, 0.006]

        # Distance between the foot rear and the foot front boxes
        boxes_distance = 0.00225

        # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = [box_size[0] + boxes_distance / 2, box_size[1] / 2, 0]
        FR_vertex_pos = [box_size[0] + boxes_distance / 2, - box_size[1] / 2, 0]
        BL_vertex_pos = [- box_size[0] - boxes_distance / 2, box_size[1] / 2, 0]
        BR_vertex_pos = [- box_size[0] - boxes_distance / 2, - box_size[1] / 2, 0]

    else:
        raise Exception("Feet vertices positions only defined for ergoCubV1.")

    # Vertices positions in the foot (F) frame
    F_vertices_pos = [FL_vertex_pos, FR_vertex_pos, BL_vertex_pos, BR_vertex_pos]

    return F_vertices_pos

def define_frontal_base_direction(robot: str) -> List:
    """Define the robot-specific frontal base direction in the base frame."""

    if robot == "ergoCubV1":
        # For ergoCubV1, the x axis is pointing forward
        frontal_base_direction = [1, 0, 0]

    else:
        raise Exception("Frontal base direction only defined for ergoCubV1.")

    return frontal_base_direction

def define_frontal_chest_direction(robot: str) -> List:
    """Define the robot-specific frontal chest direction in the chest frame."""

    if robot == "ergoCubV1":
        # For ergoCubV1, the x axis of the chest frame is pointing forward
        frontal_chest_direction = [1, 0, 0]

    else:
        raise Exception("Frontal chest direction only defined for ergoCubV1.")

    return frontal_chest_direction

def define_initial_nn_X(robot: str) -> List:
    """Define the robot-specific initial input X for the network used for trajectory generation."""

    if robot == "ergoCubV1":

        initial_nn_X = [[# Ground base positions (24)
            0.21126064124455635, -0.00036713260010769714, 0.20886300266372512, -0.0003695924081135167,
            0.20645402565381318, -0.0003604535634677853, 0.20641890607622465, -0.00035515558918680835,
            0.20967029739580362, -0.0003621222818047325, 0.0, 0.0,
            -0.2291653083856797, 0.0010798921072872226, -0.2490492176174459, 0.001080339203097057,
            -0.2490138352885897, 0.002822216647651233, -0.24130546511880147, 8.714659150120763e-05,
            -0.23340433984614908, -4.796114865371771e-07, -0.22864370874296483, 0.000645064549267912,
            # Ground facing directions (24)
            0.4942150348748153, 1.5133900321084313e-06, 0.4907375950989224, -9.140217967301278e-07,
            0.4918760972094149, -1.5019216408211414e-06, 0.4942969325957089, -1.1145506688869935e-06,
            0.4939471682166567, -6.312046913620055e-07, 0.0, 0.0020754249732434908,
            0.5259997815485239, 0.020069515503359888, 0.5146907164418114, 0.014284730935538524,
            0.5074869616535806, 0.011855784066191505, 0.5008182265015013, 0.007707347421352704,
            0.49861448577599665, 0.002920201727751729, 0.5035276727921664, 1.3399798393512078e-08,
            # Ground base velocities (24)
            -0.14683308311598753, 0.0023920527557131795, -0.15050320266361372, 0.0024421352650474686,
            -0.14525239867601603, 0.0024587738844478895, -0.14009384314533507, 0.0024911239816431405,
            -0.14552398904754127, 0.002567335280773621, -0.15893219315154467, 0.002681299111475463,
            -0.19629527216421233, 0.003596055954201884, -0.19998472636769163, 0.008360583734743939,
            -0.18236199239913176, -0.006087151029078687, -0.16431341159185325, -0.005091802408460015,
            -0.15466436930790045, 0.0014520372621927087, -0.1607691907814692, 0.0003710887323113083,
            # Joint positions (26)
            -1.0150930881500244, -0.3602990508079529, 0.3601212501525879, 0.6996650099754333,
            0.3355633616447449, 0.37254834175109863, -1.0208966732025146, -0.3418806791305542,
            0.44108936190605164, 0.6801648139953613, 0.31151413917541504, 0.3625253736972809,
            -2.051633834838867, -0.03015667200088501, 0.008174650371074677, -0.7161245346069336,
            -0.04534905403852463, 0.07089600712060928, 0.5651403069496155, -0.8951823115348816,
            0.5697140097618103, -0.6732834577560425, 0.5785202980041504, -0.8636950850486755,
            0.558005154132843, -0.6810769438743591,
            # Joint velocities (26)
            -0.03084646910429001, -0.011367272585630417, 0.01152132824063301, 0.007485941052436829,
            -0.007818274199962616, 0.013369593769311905, -0.028880421072244644, -0.00601327046751976,
            -0.005391113460063934, 0.028928734362125397, 0.01632467843592167, 8.066743612289429e-05,
            -0.21497130393981934, 0.0011840909719467163, -0.025704368948936462, -0.01040942594408989,
            -0.04307703673839569, 0.0638410672545433, 0.05322853475809097, -0.03271743282675743,
            0.011602410115301609, -0.07073163241147995, 0.09000563621520996, -0.024635441601276398,
            0.029135536402463913, -0.13268116116523743,]]

    else:
        raise Exception("Initial network input X only defined for ergoCubV1.")

    return initial_nn_X

def define_initial_past_trajectory(robot: str) -> (List, List, List):
    """Define the robot-specific initialization of the past trajectory data used for trajectory generation."""

    # The above quantities are expressed in the frame specified by the initial base position and the facing direction

    if robot == "ergoCubV1":

        # Initial past base positions manually retrieved from a standing pose
        initial_past_trajectory_base_pos = [[-2.4525983213717867e-07, -6.975809882328114e-08] ,
                                            [-2.2344327436956335e-07, -9.263491141171715e-08] ,
                                            [-1.9436342131939362e-07, -1.0783999998152734e-07] ,
                                            [-1.6250258783527665e-07, -1.129883447513048e-07] ,
                                            [-1.395474574062724e-07, -1.2642104945484442e-07] ,
                                            [-1.2891703586607023e-07, -1.266086298747296e-07] ,
                                            [-8.968030008512871e-08, -1.3588544299533094e-07] ,
                                            [-7.841528569061904e-08, -1.4050622487811873e-07] ,
                                            [-5.88352087820279e-08, -1.4417300928131768e-07] ,
                                            [-6.247196650064269e-08, -1.4967619575544551e-07] ,
                                            [-4.763094875020654e-08, -1.480676822362217e-07] ,
                                            [-2.8290643082631496e-08, -1.405078550644637e-07] ,
                                            [-8.140976336603932e-09, -1.43048605709479e-07] ,
                                            [5.209698495265807e-09, -1.350494365968052e-07] ,
                                            [1.864453856157105e-08, -1.2640264928471784e-07] ,
                                            [1.8934465781228015e-08, -1.2745781869056638e-07] ,
                                            [3.3126552538519095e-08, -1.1514297751649374e-07] ,
                                            [4.104176328290591e-08, -1.0716085961258652e-07] ,
                                            [6.20561652084999e-08, -1.0311402426797963e-07] ,
                                            [5.6066621728020606e-08, -1.0357286708594721e-07] ,
                                            [5.639345534150664e-08, -1.022369471264267e-07] ,
                                            [5.5525441440650283e-08, -9.264156173389834e-08] ,
                                            [5.0674382458296524e-08, -8.275811232536074e-08] ,
                                            [6.519947796512609e-08, -7.337540758891175e-08] ,
                                            [5.326112450548404e-08, -6.817576405331165e-08] ,
                                            [6.842072538892837e-08, -6.232807382721999e-08] ,
                                            [5.516487038001275e-08, -5.6970218849720976e-08] ,
                                            [7.087872544041648e-08, -5.247710784422901e-08] ,
                                            [5.6728111749016084e-08, -4.632158356623371e-08] ,
                                            [5.0883313344129744e-08, -4.5062546451360656e-08] ,
                                            [4.513758291792905e-08, -3.5279611383150206e-08] ,
                                            [4.555607708297271e-08, -2.9807026162377406e-08] ,
                                            [3.9106534285458154e-08, -2.9703673049173513e-08] ,
                                            [4.7808490130891986e-08, -2.004964294741561e-08] ,
                                            [3.4316054830500126e-08, -1.8749961176271295e-08] ,
                                            [5.0151156744370066e-08, -1.3014513820728749e-08] ,
                                            [4.344268039262971e-08, -7.891402351122915e-09] ,
                                            [4.4337264340313625e-08, -6.8927016756917005e-09] ,
                                            [4.53235959221356e-08, -2.6555198366506486e-09] ,
                                            [4.5560088767797906e-08, 2.3008390105250562e-09] ,
                                            [3.141573798549465e-08, 3.2901223306188416e-09] ,
                                            [3.2952322350554286e-08, 4.242855719185823e-09] ,
                                            [2.6804188352284173e-08, 4.64151541796809e-09] ,
                                            [1.924697985505423e-08, 1.6353611339495619e-10] ,
                                            [5.609624867518265e-09, 1.5417095712728314e-09] ,
                                            [2.0452219745623622e-08, 7.308031660759899e-09] ,
                                            [5.8224449825230744e-09, 7.818895378908178e-09] ,
                                            [1.376732887981793e-08, 1.200850085857676e-08] ,
                                            [1.4517080315177495e-08, 1.21875084631999e-08] ,
                                            [1.4607508533872508e-08, 7.668884614716481e-09] ,
                                            [0.0, 0.0]]

        # Initial past facing directions manually retrieved from a standing pose
        initial_past_trajectory_facing_dirs = [[0.9999999999999992, 2.9093091923590025e-08] ,
                                                [0.9999999999999994, 2.014911975584149e-08] ,
                                                [0.9999999999999999, 1.3179330542341244e-08] ,
                                                [0.9999999999999998, 6.647496607772829e-09] ,
                                                [1.0, -4.873720972419595e-10] ,
                                                [0.9999999999999997, -4.7268188579709805e-09] ,
                                                [0.9999999999999999, -1.0313678423231605e-08] ,
                                                [0.9999999999999999, -1.5392446914458143e-08] ,
                                                [0.9999999999999994, -1.82230847293005e-08] ,
                                                [0.9999999999999992, -2.5280323210961714e-08] ,
                                                [0.9999999999999999, -2.7486334006365693e-08] ,
                                                [0.9999999999999998, -3.301822727891429e-08] ,
                                                [0.9999999999999998, -3.478949768439227e-08] ,
                                                [1.0000000000000002, -3.593399761419225e-08] ,
                                                [0.999999999999999, -3.796850549693033e-08] ,
                                                [1.0, -3.9978205957751573e-08] ,
                                                [0.9999999999999984, -3.904179780760617e-08] ,
                                                [0.9999999999999997, -3.754982677612068e-08] ,
                                                [0.9999999999999993, -3.5549764562680984e-08] ,
                                                [1.0, -3.5454821146051464e-08] ,
                                                [0.999999999999999, -3.350651660429064e-08] ,
                                                [0.9999999999999992, -3.271680955859227e-08] ,
                                                [0.9999999999999992, -3.194762391867564e-08] ,
                                                [0.9999999999999989, -3.3709256069708154e-08] ,
                                                [0.9999999999999991, -3.1828944614467964e-08] ,
                                                [0.9999999999999988, -3.011692712527426e-08] ,
                                                [0.9999999999999989, -2.717510452336319e-08] ,
                                                [0.9999999999999992, -2.618265803479871e-08] ,
                                                [0.9999999999999992, -2.447064056803139e-08] ,
                                                [0.9999999999999991, -2.389241328611421e-08] ,
                                                [0.9999999999999996, -2.210704526790347e-08] ,
                                                [0.9999999999999994, -2.1044462685288662e-08] ,
                                                [0.9999999999999996, -1.9862129293008285e-08] ,
                                                [0.9999999999999997, -1.6505016049498375e-08] ,
                                                [0.9999999999999998, -1.5396033238852374e-08] ,
                                                [0.9999999999999996, -1.5633391920251134e-08] ,
                                                [0.9999999999999999, -1.3387404474592523e-08] ,
                                                [0.9999999999999997, -1.2323750423983074e-08] ,
                                                [0.9999999999999999, -8.895429593024653e-09] ,
                                                [0.9999999999999998, -7.298876996907437e-09] ,
                                                [0.9999999999999998, -4.910474346202517e-09] ,
                                                [0.9999999999999997, -3.918027891631847e-09] ,
                                                [0.9999999999999998, -3.4574084631499134e-09] ,
                                                [0.9999999999999998, -2.9277244970831855e-09] ,
                                                [0.9999999999999999, -2.4217764023647915e-09] ,
                                                [0.9999999999999999, -3.1672262515143825e-09] ,
                                                [0.9999999999999999, -2.080907882427666e-09] ,
                                                [1.0, 8.155859188058726e-10] ,
                                                [0.9999999999999999, 6.753137875760107e-10] ,
                                                [0.9999999999999999, 7.475929410310381e-10] ,
                                                [1.0, 0.0]]

        # Initial past base velocities manually retrieved from a standing pose
        initial_past_trajectory_base_vel = [[0.02893717126394264, 0.018655631852214995] ,
                                            [0.028937204330499414, 0.01865576250545683] ,
                                            [0.028937248409131044, 0.01865593523098602] ,
                                            [0.028937265617299875, 0.018656099267057937] ,
                                            [0.02893727755450549, 0.018656276648929767] ,
                                            [0.0289372838719281, 0.018656416358233643] ,
                                            [0.028937268726345857, 0.018656553995989453] ,
                                            [0.02893728549469845, 0.018656653704554] ,
                                            [0.02893729685559311, 0.01865674679520928] ,
                                            [0.028937292039612213, 0.018656826686706526] ,
                                            [0.028937298084045406, 0.018656926468592006] ,
                                            [0.028937314655238983, 0.018656997340687688] ,
                                            [0.02893729932766382, 0.018657108360164588] ,
                                            [0.028937299979805185, 0.018657203742330715] ,
                                            [0.028937289817030663, 0.01865728588867839] ,
                                            [0.0289372743226286, 0.01865737250806629] ,
                                            [0.028937269324654808, 0.018657425781284598] ,
                                            [0.028937248407628763, 0.018657503564573356] ,
                                            [0.02893723809319346, 0.01865756352902189] ,
                                            [0.02893722774842602, 0.018657619057090607] ,
                                            [0.028937227930418974, 0.018657645675369507] ,
                                            [0.028937222811116516, 0.018657681203068568] ,
                                            [0.028937206983060765, 0.018657719022278453] ,
                                            [0.028937191139838933, 0.018657754623298467] ,
                                            [0.028937191246001488, 0.018657770150627806] ,
                                            [0.028937153803280095, 0.018657783716390604] ,
                                            [0.028937148608147254, 0.0186578081531401] ,
                                            [0.02893713256776639, 0.01865781491769127] ,
                                            [0.0289371219196774, 0.018657826081961752] ,
                                            [0.028937100532502955, 0.018657835101363304] ,
                                            [0.028937100623499416, 0.01865784841050276] ,
                                            [0.028937100684163726, 0.018657857283262424] ,
                                            [0.028937095322204034, 0.018657857319922883] ,
                                            [0.028937089869247888, 0.018657844047443903] ,
                                            [0.028937079145328497, 0.018657844120764842] ,
                                            [0.0289370790694981, 0.018657833029815305] ,
                                            [0.028937079114996336, 0.018657839684385027] ,
                                            [0.028937073631708013, 0.018657821975526217] ,
                                            [0.02893705748516463, 0.01865781321274801] ,
                                            [0.028937057424500316, 0.018657804339988367] ,
                                            [0.02893705201704239, 0.018657797722079116] ,
                                            [0.028937052017042386, 0.01865779772207912] ,
                                            [0.02893704656408623, 0.018657784449600135] ,
                                            [0.02893704642759153, 0.018657764485890946] ,
                                            [0.02893702487359021, 0.018657749105203486] ,
                                            [0.02893701935996974, 0.018657726959964847] ,
                                            [0.028937013952511807, 0.01865772034205559] ,
                                            [0.02893701923864112, 0.01865770921444557] ,
                                            [0.028937024600600814, 0.018657709177785097] ,
                                            [0.028937024479272187, 0.018657691432265814] ,
                                            [0.028937018935319556, 0.01865766485064737]]

    else:
        raise Exception("Initial past trajectory data only defined for ergoCubV1.")

    return initial_past_trajectory_base_pos, initial_past_trajectory_facing_dirs, initial_past_trajectory_base_vel

def define_initial_base_height(robot: str) -> List:
    """Define the robot-specific initial height of the base frame."""

    if robot == "ergoCubV1":
        initial_base_height = 0.7748

    else:
        raise Exception("Initial base height only defined for ergoCubV1.")

    return initial_base_height

def define_initial_support_foot_and_vertex(robot: str) -> List:
    """Define the robot-specific initial support foot and vertex."""

    if robot == "ergoCubV1":
        initial_support_foot = "left_foot"
        initial_support_vertex = 4

    else:
        raise Exception("Initial support foot and vertex only defined for ergoCubV1.")

    return initial_support_foot, initial_support_vertex

def define_initial_base_yaw(robot: str) -> List:
    """Define the robot-specific initial base yaw expressed in the world frame."""

    if robot == "ergoCubV1":
        # For ergoCubV1, the initial base yaw is 0 degs since the x axis of the base frame points forward
        initial_base_yaw = 0

    else:
        raise Exception("Initial base yaw only defined for ergoCubV1.")

    return initial_base_yaw

def define_initial_feet_positions(robot: str) -> (List, List):
    """Define the robot-specific initial positions of the feet frames."""

    if robot == "ergoCubV1":
        l_foot_position = [0, 0.08, 0]
        r_foot_position = [0, -0.08, 0]

    else:
        raise Exception("Initial feet position only defined for ergoCubV1.")

    return l_foot_position, r_foot_position

# ===================
# VISUALIZATION UTILS
# ===================

from PIL import ImageColor
from matplotlib.patches import Polygon

def to_rgb(color, alpha=1.0):
    c = [v / 255.0 for v in ImageColor.getcolor(color, "RGB")]
    c.append(alpha)
    return c

def plot_rotated_rectangle(width, height, origin, quaternion, color, linewidth, label=None):

    from matplotlib.patches import Polygon
    from scipy.spatial.transform import Rotation

    # Extract quaternion components
    w, x, y, z = quaternion

    # Compute rotation matrix from the quaternion (w, x, y, z) to (x, y, z, w) convention
    rotation_matrix = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2]
    ])

    # Define the vertices of the rectangle centered at the specified origin
    half_width = width / 2
    half_height = height / 2
    vertices = np.array([[-half_width, -half_height],
                         [half_width, -half_height],
                         [half_width, half_height],
                         [-half_width, half_height]])


    # Rotate the vertices using the rotation matrix
    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    # Translate the vertices to the specified origin
    translated_vertices = rotated_vertices + np.array(origin)

    # Create a new Rectangle object with the correct coordinates
    if label is None:
        rectangle = Polygon(translated_vertices, fill=True, color=color, alpha=0.7, edgecolor=color, linewidth=linewidth)
    else:
        rectangle = Polygon(translated_vertices, fill=True, color=color, alpha=0.7, edgecolor=color, label=label, linewidth=linewidth)

    return rectangle


def visualize_generated_motion(icub: iCub,
                               controlled_joints_indexes: List,
                               gazebo: scenario.GazeboSimulator,
                               l_footsteps: Dict,
                               r_footsteps: Dict,
                               posturals: Dict,
                               raw_data: List,
                               blending_coeffs: Dict,
                               plot_blending_coeffs: bool = False,
                               plot_joystick_inputs: bool = False,
                               plot_com: bool = False,
                               plot_momentum: bool = False) -> None:
    """Visualize the generated motion, optionally along with the joystick inputs used to generate it, the activations
    of the blending coefficients, the com and the momentum evolution during the trajectory generation."""

    # Extract posturals
    joint_pos_posturals = posturals["joints_pos"]
    joint_vel_posturals = posturals["joints_vel"]
    base_posturals = posturals["base"]
    com_pos_posturals = posturals["com_pos"]
    com_vel_posturals = posturals["com_vel"]
    centroidal_momentum_posturals = posturals["centroidal_momentum"]

    # Define controlled joints
    icub_joints = icub.joint_names()

    # Config before any plot
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    mpl.rcParams['toolbar'] = 'None'
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 25})

    print("Matplotlib backend:", mpl.get_backend())

    # Plot configuration
    plt.ion()

    for frame_idx in range(len(joint_pos_posturals)):

        # Debug
        print(frame_idx, "/", len(joint_pos_posturals))

        # Plot configuration
        xticks = [i for i in range(0, frame_idx, 300)]
        xtick_labels = [str(i / 300) for i in range(0, frame_idx, 300)]

        # ======================
        # VISUALIZE ROBOT MOTION
        # ======================

        # Retrieve the current joint positions
        joint_postural = joint_pos_posturals[frame_idx]

        full_joint_positions = np.zeros(len(icub_joints))
        for index in controlled_joints_indexes:
            full_joint_positions[index] = joint_postural[icub_joints[index]]

        # Retrieve the current base position and orientation
        base_postural = base_posturals[frame_idx]
        base_position = base_postural['position']
        base_quaternion = base_postural['wxyz_quaternions']

        # Reset the robot configuration in the simulator
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(full_joint_positions, icub_joints)
        gazebo.run(paused=True)

        # =====================================
        # PLOT THE MOTION DIRECTION ON FIGURE 1
        # =====================================

        if plot_joystick_inputs:

            # Retrieve the current motion direction
            curr_raw_data = raw_data[frame_idx]
            curr_x = curr_raw_data[0]
            curr_y = curr_raw_data[1]

            plt.figure(1, figsize=(8, 8))
            plt.clf()

            # Circumference of unitary radius
            r = 1
            x = np.linspace(-r, r, 1000)
            y = np.sqrt(-x ** 2 + r ** 2)
            plt.plot(x, y, 'r', linewidth=5.0)
            plt.plot(x, -y, 'r', linewidth=5.0)

            # Motion direction
            plt.scatter(0, 0, c='r')
            desired_motion_direction = plt.arrow(0, 0, curr_x, -curr_y, length_includes_head=True, width=0.02,
                                                 head_width=8 * 0.01, head_length=1.8 * 8 * 0.01, color='r')

            # Plot configuration
            plt.axis('scaled')
            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.4, 1.2])
            plt.axis('off')
            plt.legend([desired_motion_direction], ['User input'], loc="lower center")

        # ==========================
        # PLOT COM POS AND FOOTSTEPS
        # ==========================

        if plot_com:

            color_left_nominal = [0.4660, 0.6740, 0.1880, 0.8]
            color_left_nominal = to_rgb("#b2e061", 0.9)
            color_right_nominal = color_left_nominal
            color_com_nominal = to_rgb("#9b19f5")

            # Retrieve the com pos posturals up to the current time
            com_pos_postural = com_pos_posturals[:frame_idx]
            com_pos_postural_x = [elem[1] for elem in com_pos_postural]
            com_pos_postural_y = [-elem[0] for elem in com_pos_postural]

            plt.figure(4, figsize=(16, 24))
            plt.clf()

            plt.plot(com_pos_postural_x, com_pos_postural_y, label='CoM nominal', color=color_com_nominal, linewidth=5.0)

            for element in l_footsteps:
                if element["activation_time"] < frame_idx / 100.0:
                    ground_l_footstep_x = element["pos"][1]
                    ground_l_footstep_y = -element["pos"][0]
                    l_quat = element["quat"]

                    if element["activation_time"] == 0:
                        rectangle = plot_rotated_rectangle(0.06,0.16, (ground_l_footstep_x, ground_l_footstep_y), l_quat, color=color_left_nominal, label='footsteps nominal', linewidth=5.0)
                    else:
                        rectangle = plot_rotated_rectangle(0.06, 0.16, (ground_l_footstep_x, ground_l_footstep_y), l_quat, color=color_left_nominal, linewidth=5.0)
                    plt.gca().add_patch(rectangle)

            for element in r_footsteps:
                if element["activation_time"] < frame_idx / 100.0:
                    ground_r_footstep_x = element["pos"][1]
                    ground_r_footstep_y = -element["pos"][0]
                    r_quat = element["quat"]

                    if element["activation_time"] == 0:
                        rectangle = plot_rotated_rectangle(0.06,0.16, (ground_r_footstep_x, ground_r_footstep_y), r_quat, color=color_right_nominal, label=None, linewidth=5.0)
                    else:
                        rectangle = plot_rotated_rectangle(0.06, 0.16, (ground_r_footstep_x, ground_r_footstep_y), r_quat, color=color_right_nominal, linewidth=5.0)
                    plt.gca().add_patch(rectangle)

            # Plot configuration
            plt.axis('scaled')
            plt.xlim(-0.8,1.05)
            plt.ylim(-1.75,0.2)
            plt.ylabel("y (m)")
            plt.xlabel("x (m)")
            plt.legend(loc='lower center', ncol=3)

        if plot_joystick_inputs or plot_com:
            plt.show()
            plt.pause(0.001)
        else:
            # Show robot motion in real time
            time.sleep(0.01)

        if frame_idx == 1:
            input("Start video")

    input("Press Enter to end the visualization of the generated trajectory.")


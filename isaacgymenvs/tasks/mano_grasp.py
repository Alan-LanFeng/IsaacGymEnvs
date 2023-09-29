# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import pickle
from isaacgym import gymtorch
from isaacgym import gymapi
import trimesh
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp
from isaacgymenvs.utils.mano_utils import *
from tqdm import tqdm
from manopth.manolayer import ManoLayer
from isaacgymenvs.tasks.base.vec_task import VecTask
import random

def write_urdf(path):
    with open(os.path.join(path,'ige.urdf'),'w') as f:
        f.write('?xml version="1.0"?>\n')
        f.write('<robot name="object">\n')
        f.write('<link name="object">\n')
        f.write('<visual>\n')
        f.write('<origin xyz="0 0 0"/>\n')
        f.write('<geometry>\n')
        f.write('<mesh filename="textured_simple.obj" scale="1 1 1"/>\n')
        f.write('</geometry>\n')
        f.write('</visual>\n')
        f.write('<collision>\n')
        f.write('<origin xyz="0 0 0"/>\n')
        f.write('<geometry>\n')
        f.write('<mesh filename="textured_simple.obj" scale="1 1 1"/>\n')
        f.write('</geometry>\n')
        f.write('</collision>\n')
        f.write('<inertial>\n')
        f.write('<density value="567.0"/>\n')
        f.write('</inertial>\n')
        f.write('</link>\n')
        f.write('</robot>\n')
    return


class obj_asset_isaac():
    def __init__(self, root,qpos_reset, qpos_ref,ee_ref,contact_ref,table_pose,sample_num=100):
        self.root = root
        obj_mesh = trimesh.load(root + '/textured_simple.obj')
        obj_size = obj_mesh.bounds[1] - obj_mesh.bounds[0]
        self.max_size = max(obj_size)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = table_pose.p.x
        object_start_pose.p.y = table_pose.p.y
        object_start_pose.p.z = table_pose.p.z*2-obj_mesh.bounds[0][2]
        self.object_start_pose = object_start_pose

        obj_pos = [object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z]

        qpos_reset[:3]+=obj_pos
        qpos_ref[:3]+=obj_pos
        ee_ref[:]+=obj_pos

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*qpos_reset[:3])
        reset_euler = qpos_reset[3:6]
        reset_quat = euler_angles_to_quaternion(torch.tensor(reset_euler), 'XYZ')
        shadow_hand_start_pose.r = gymapi.Quat(*reset_quat)
        self.shadow_hand_start_pose = shadow_hand_start_pose

        verts = trimesh.sample.sample_surface(obj_mesh, sample_num)[0]
        verts = np.array(verts)
        self.verts = verts

        self.qpos_ref = qpos_ref
        self.ee_ref = ee_ref
        self.contact_ref = contact_ref
        #qpos_reset[:6]=0
        self.qpos_reset = qpos_reset
        self.object_asset = None
    def get_asset(self,gym,sim):
        if not self.object_asset:
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.vhacd_enabled = True
            object_asset_options.density = 10
            # object_asset_options.override_com = True
            # object_asset_options.override_inertia = True

            path = self.root
            # create "ige.urdf" fole in path if not exist
            if not os.path.exists(os.path.join(path,'ige.urdf')):
                # create ige.urdf
                write_urdf(path)
            object_asset = gym.load_asset(sim, path, 'ige.urdf', object_asset_options)
            self.object_asset = object_asset
            return object_asset
        else:
            return self.object_asset


class obj_aggregation_isaac():
    def __init__(self, obj_asset_list,device):
        self.obj_asset_list = obj_asset_list
        vert_list = [obj.verts for obj in obj_asset_list]
        root_list = [obj.root for obj in obj_asset_list]
        qpos_reset_list = [obj.qpos_reset for obj in obj_asset_list]
        qpos_ref_list = [obj.qpos_ref for obj in obj_asset_list]
        ee_ref_list = [obj.ee_ref for obj in obj_asset_list]
        contact_ref_list = [obj.contact_ref for obj in obj_asset_list]


        self.sample_num = obj_asset_list[0].verts.shape[0]
        self.vert_tensor = torch.from_numpy(np.stack(vert_list, axis=0)).float().to(device)
        self.qpos_reset_tensor = torch.from_numpy(np.stack(qpos_reset_list, axis=0)).float().to(device)
        self.qpos_ref_tensor = torch.from_numpy(np.stack(qpos_ref_list, axis=0)).float().to(device)
        self.ee_ref_tensor = torch.from_numpy(np.stack(ee_ref_list, axis=0)).float().to(device)
        self.contact_ref_tensor = torch.from_numpy(np.stack(contact_ref_list, axis=0)).float().to(device)
        self.root_list = root_list

    def __len__(self):
        return len(self.obj_asset_list)

class PCARegularizer():
    def __init__(self,device, thresh=-0.4,scale=0.1,):
        mano_layer = ManoLayer(mano_root='data', flat_hand_mean=False, ncomps=45, use_pca=True)
        mean_axisang = mano_layer.th_hands_mean
        self.th_comps = mano_layer.th_comps.to(device)
        self.mean_pca = torch.einsum('bi,ij->bj', [mean_axisang, torch.inverse(mano_layer.th_comps)]).to(device)
        self.scale = scale
        self.thresh = thresh

    def get_pca_rewards(self, param):
        bs = param.shape[0]

        mano_param = dgrasp_to_mano(param)
        #ref = dgrasp_to_mano_np(param.detach().cpu().numpy())

        # joint_pca = np.matmul(mano_param[:,3:48], np.linalg.inv(self.th_comps))
        # joint_pca_norm = joint_pca / np.linalg.norm(joint_pca, axis=-1, keepdims=True)
        #
        # pca_target = self.mean_pca.repeat(bs, 0)
        # pca_target_norm = pca_target / np.linalg.norm(pca_target, axis=-1, keepdims=True)
        #
        # pca_dist = joint_pca_norm * pca_target_norm
        # cos_sim = pca_dist.sum(-1)-1
        # cos_sim[cos_sim>self.thresh] *= self.scale
        # return cos_sim

        joint_pca = torch.mm(mano_param[:, 3:48], torch.inverse(self.th_comps))
        joint_pca_norm = joint_pca / torch.norm(joint_pca, dim=-1, keepdim=True)

        pca_target = self.mean_pca.repeat(bs, 1)
        pca_target_norm = pca_target / torch.norm(pca_target, dim=-1, keepdim=True)

        pca_dist = joint_pca_norm * pca_target_norm
        cos_sim = pca_dist.sum(-1)-1
        cos_sim[cos_sim > self.thresh] *= self.scale

        return cos_sim

class ManoGrasp(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.sample_num = 100
        self.reward_dict = {}
        self.cfg = cfg
        self.label = pickle.load(open(self.cfg["task"]["label_file"], "rb"))
        self.grasp_reference = pickle.load(open(self.cfg["task"]["grasp_reference"], "rb"))
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.hand_trans_stiffness = self.cfg["env"]["handTranslationStiffness"]
        self.hand_rot_stiffness = self.cfg["env"]["handRotationStiffness"]
        self.hand_joint_stiffness = self.cfg["env"]["handJointStiffness"]


        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "table": "urdf/table_narrow.urdf"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 457+self.sample_num*3
        }

        self.up_axis = 'z'
        self.handbody = [ "link_ff_pm_z", "link_ff_md_z", "link_ff_dd_z",
                        "link_mf_pm_z", "link_mf_md_z", "link_mf_dd_z",
                        "link_rf_pm_z", "link_rf_md_z", "link_rf_dd_z",
                        "link_lf_pm_z", "link_lf_md_z", "link_lf_dd_z",
                        "link_th_pm_z", "link_th_md_z", "link_th_dd_z"]

        self.hand_mount = ["link_palm_rz"]
        self.num_hand_body = len(self.handbody)
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 51
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0-9, 5.0-5.5, 0.5)
            cam_target = gymapi.Vec3(6.0-9, 5.0-5.5, 0.0)
            # cam_pos = gymapi.Vec3(10.0, 5.0, 0.5)
            # cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)


        if self.obs_type == "full_state" or self.asymmetric_obs:
            # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_hand_body * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


        self.pca_regualizer = PCARegularizer(self.device,thresh=-0.3)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_force = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, -1, 3)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.rb_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == 'z' else 1 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand.xml")

        if "asset" in self.cfg["env"]:
            # asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = os.path.normpath(self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file))


        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = False

        table_asset_options.density = 1000
        table_width, table_height, table_depth = 0.5, 0.5, 0.2

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = 0
        table_pose.p.y = -0.4
        table_pose.p.z = table_depth / 2

        table_asset = self.gym.create_box(self.sim, table_width, table_height, table_depth, table_asset_options)

        grasp_reference = [v for k,v in self.grasp_reference.items()]

        obj_num=20

        obj_list = []
        rand_idx = np.random.choice(len(grasp_reference), obj_num, replace=False)
        for idx in rand_idx:
            root = '/media/lafeng/81bf96d9-51d5-4fce-b089-c1285da333d21/mp_v3'
            ref = grasp_reference[idx]

            obj_pos = ref['obj_pose_reset'][0,:3]
            qpos_ref = ref['final_qpos'][0]
            qpos_ref[0:3] -= obj_pos
            ee_ref = ref['final_ee'][0]
            ee_ref = ee_ref.reshape(-1,3)-obj_pos.reshape(-1,3)
            contact_ref = ref['final_contacts'][0]
            qpos_reset = ref['qpos_reset'][0]
            qpos_reset[0:3] -= obj_pos

            obj_name = grasp_reference[idx]['obj_name'][0]
            root = root+'/'+obj_name
            obj = obj_asset_isaac(root,qpos_reset,qpos_ref,ee_ref,contact_ref,table_pose, self.sample_num)
            if obj.max_size > 0.15:
                continue
            obj_list.append(obj)

        obj_list= obj_list[:2]
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.use_mesh_materials = True
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        m = self.gym.get_asset_rigid_shape_properties(shadow_hand_asset)
        for prop in m:
            prop.friction = 5.0
        self.gym.set_asset_rigid_shape_properties(shadow_hand_asset, m)
        #m = self.gym.get_asset_rigid_shape_properties(shadow_hand_asset)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)


        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]
        self.actuated_dof_indices = np.array(range(self.num_shadow_hand_dofs))
        # get shadow_hand dof properties, loaded by Isaac Gym from the MJCF file
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        shadow_hand_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        shadow_hand_dof_props["stiffness"][:3] = self.hand_trans_stiffness
        shadow_hand_dof_props["stiffness"][3:6] = self.hand_rot_stiffness
        shadow_hand_dof_props["stiffness"][6:] = self.hand_joint_stiffness

        shadow_hand_dof_props["damping"]= shadow_hand_dof_props["stiffness"] * 0.1

        shadow_hand_dof_props["velocity"].fill(100)
        shadow_hand_dof_props["effort"].fill(100)

        # get shadow_hand dof limits
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []


        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.int32, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        self.hand_body_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in
                                  self.handbody]


        # rotate the hand so that the palm faces the object
        # r1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.pi/2)
        # r2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), np.pi/2)
        # shadow_hand_start_pose.r = r2 * r1

        # load table asset

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies
        max_agg_shapes = self.num_shadow_hand_shapes

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []
        self.table_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.object_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + 1))

        self.hand_mount_handle = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in
                                  self.hand_mount]

        self.object_rb_masses = []

        sampled_obj = []
        for i in tqdm(range(self.num_envs)):

            # randomize h
            obj_idx = random.randint(0, len(obj_list)-1)
            obj = obj_list[obj_idx]
            sampled_obj.append(obj)
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            object_asset = obj.get_asset(self.gym,self.sim)
            agg_bodies = max_agg_bodies + self.gym.get_asset_rigid_body_count(object_asset)
            agg_shapes = max_agg_shapes + self.gym.get_asset_rigid_shape_count(object_asset)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, agg_bodies, agg_shapes, True)


            shadow_hand_start_pose = obj.shadow_hand_start_pose
            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0b01, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # enable DOF force sensors, if needed
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)


            object_start_pose = obj.object_start_pose
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0b00, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            self.object_rb_masses += [prop.mass for prop in object_rb_props]

            table_pose.p = gymapi.Vec3()
            table_pose.p.x = 0
            table_pose.p.y = -0.4
            table_pose.p.z = table_depth / 2
            self.table_start_states.append([table_pose.p.x, table_pose.p.y, table_pose.p.z,
                                             table_pose.r.x, table_pose.r.y, table_pose.r.z, table_pose.r.w,
                                             0, 0, 0, 0, 0, 0])
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0b01, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)


            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.obj_aggregation = obj_aggregation_isaac(sampled_obj, device=self.device)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        self.obj_verts = self.obj_aggregation.vert_tensor

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.table_start_states = to_torch(self.table_start_states, device=self.device).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.int32, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.int32, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.int32, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.int32, device=self.device)

    def compute_reward(self, actions):

        # self.rew_buf[:], self.reset_buf[:], reward_dict, self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
        #     self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
        #     self.max_episode_length, self.object_pos, self.object_rot, self.object_linvel,self.object_angvel,self.object_init_state, self.hand_pos,
        #     self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
        #     self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
        #     self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"),self.contact_force[:,self.hand_body_handles],self.contact_force[:,self.object_rb_handles+1]
        # )

        obj_linevel = self.object_linvel
        obj_angvel = self.object_angvel
        progress_buf = self.progress_buf
        max_episode_length = self.max_episode_length
        reset_buf = self.reset_buf
        finger_force = self.contact_force[:, self.hand_body_handles]
        #table_force = self.contact_force[:, self.object_rb_handles+1]
        #obj_force = self.contact_force[:, self.object_rb_handles]
        default_obj_force = self.object_rb_masses * 9.81

        object_pos = self.object_pos

        reward_dict = {}
        action_penalty = torch.sum(actions ** 2, dim=-1)
        obj_vel_penalty = torch.sum(obj_linevel ** 2, dim=-1) + torch.sum(obj_angvel ** 2, dim=-1)
        resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

        impulse = torch.norm(finger_force[..., :3], dim=-1).sum(-1)# - torch.norm(table_force[..., [2]], dim=-1).sum(-1)  # + 50*9.81
        impulse = torch.clamp(impulse, min=-default_obj_force, max=default_obj_force*3)

        obj_init_height = self.object_init_state[..., 2].clone()
        obj_height = object_pos[..., 2] - obj_init_height
        height_reward = torch.clamp(obj_height, min=0, max=0.5)

        # obj_movement = torch.norm(object_pos-obj_init_state[...,:3],dim=-1)
        # object_rot[:] = obj_init_state[...,3:7] = 0.5
        # quat_diff = quat_mul(object_rot, quat_conjugate(obj_init_state[...,3:7]))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        # hand_dof = self.shadow_hand_dof_pos.clone()
        # pca_reward = self.pca_regualizer.get_pca_rewards(hand_dof)

        pca_reward = 0
        scale = {}
        scale['action_penalty'] = -0.3
        scale['obj_vel_penalty'] = -0.5
        scale['height_reward'] = 0
        scale['impulse'] = 3
        scale['pca_reward'] = 2

        reward = (action_penalty * scale['action_penalty'] + obj_vel_penalty * scale['obj_vel_penalty']
                  + height_reward * scale['height_reward'] + impulse * scale['impulse']) # pca_reward * scale['pca_reward'])

        reward_dict['action_penalty'] = action_penalty.mean().item() * scale['action_penalty']
        reward_dict['obj_vel_penalty'] = obj_vel_penalty.mean().item() * scale['obj_vel_penalty']
        reward_dict['height_reward'] = height_reward.mean().item() * scale['height_reward']
        reward_dict['impulse'] = impulse.mean().item() * scale['impulse']
        #reward_dict['pca_reward'] = pca_reward.mean().item() * scale['pca_reward']
        reward_dict['reward'] = reward.mean().item()

        self.rew_buf[:], self.reset_buf[:], reward_dict = reward, resets, reward_dict


        cur_step = self.progress_buf[0].item()
        if cur_step == 1:
            self.reward_dict = reward_dict
        else:
            for k in reward_dict.keys():
                self.reward_dict[k] += reward_dict[k]
        if cur_step == self.max_episode_length-1:
            for k in reward_dict.keys():
                self.reward_dict[k]/=self.max_episode_length

                self.extras['reward_info'] = reward_dict


    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            #self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        obj_info = torch.cat([self.object_pos,self.object_rot,self.object_linvel,self.object_angvel],dim=1)
        obj_info = self.transform_object_to_hand_frame(obj_info)
        obj_pos = obj_info[:,0:3]
        obj_rot = obj_info[:,3:7]
        # transform self.obj_verts


        verts = self.obj_verts.clone()
        verts = rotate_pcd_with_quaternion(verts, obj_rot)
        self.verts = verts + obj_pos.unsqueeze(1)
        self.verts = self.verts.reshape(self.num_envs, -1)

        # if self.progress_buf[0].item() in range(0,100,10):
        #     hand_dof = self.shadow_hand_dof_pos
        #     mano_param = dgrasp_to_mano(hand_dof.cpu().numpy())
        #     mano_param = torch.from_numpy(mano_param).float()
        #     mano_layer = ManoLayer(mano_root='data', use_pca=False)
        #     mano_verts,joints = mano_layer(th_pose_coeffs=mano_param[:, :48], th_trans=mano_param[:, 48:])
        #     mano_verts = mano_verts.detach().numpy() / 1000
        #     for i in range(1):
        #         show_pointcloud_objhand(mano_verts[i],verts[i].cpu().detach().numpy())


        self.hb_state = self.rigid_body_states[:, self.hand_body_handles][:, :, 0:13]
        self.hb_pos = self.rigid_body_states[:, self.hand_body_handles][:, :, 0:3]
        self.hand_pos = self.rigid_body_states[:,self.hand_mount_handle,:3].squeeze(1)

        self.compute_full_state()


    def compute_full_state(self, asymm_obs=False):

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

        obj_obs_start = 3*self.num_shadow_hand_dofs  # 153

        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
        self.obs_buf[:,obj_obs_start:obj_obs_start+13] = self.transform_object_to_hand_frame(self.obs_buf[:,obj_obs_start:obj_obs_start+13])

        # fingertip observations, state(pose and vel) + force-torque sensors
        num_hb_states = 13 * self.num_hand_body  # 208
        num_hb_force_torques = 3 * self.num_hand_body  # 48

        hb_obs_start = obj_obs_start + 13
        self.obs_buf[:, hb_obs_start:hb_obs_start + num_hb_states] = self.hb_state.reshape(self.num_envs, num_hb_states)
        self.obs_buf[:, hb_obs_start + num_hb_states:hb_obs_start + num_hb_states +
                     num_hb_force_torques] = self.force_torque_obs_scale * self.contact_force[:,self.hand_body_handles].reshape(self.num_envs,-1)

        # obs_end = 470
        # obs_total = obs_end + num_actions = 521
        obs_end = hb_obs_start + num_hb_states + num_hb_force_torques
        self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions
        self.obs_buf[:, obs_end + self.num_actions:] = self.verts

    def reset_idx(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        self.rb_torques[env_ids, :, :] = 0.0


        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_shadow_hand_dofs] + 1)
        qpos_reset = self.obj_aggregation.qpos_reset_tensor.clone()
        qpos_reset[:,:6]=0
        pos = qpos_reset + self.reset_dof_pos_noise * rand_delta

        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        obj_indices = self.object_indices[env_ids].to(torch.int32)
        table_indices = self.table_indices[env_ids]
        all_indices = torch.unique(torch.cat([hand_indices,
                                              obj_indices,table_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(hand_indices))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        # reset object
        self.root_state_tensor[obj_indices] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[table_indices] = self.table_start_states[env_ids].clone()
        self.root_state_tensor[self.hand_indices[env_ids]] = self.hand_start_states[env_ids].clone()
        #self.root_state_tensor[:] = self.root_state_tensor[1]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))


        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)


        if self.use_relative_control:
            add = self.actions.clone()*5e-2
            add[:,:3]*=0.02
            add[:, 3:6]*=0.1
            add[:,6:]*=0.5
            targets = self.prev_targets[:, self.actuated_dof_indices] + add
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:

            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])



        wrist_pos_final = self.obj_aggregation.qpos_ref_tensor[:,:3]

        # obj_pos = self.hand_start_states[:,:3].clone()
        # obj_pos[:,2]+=0.1
        wrist_pos_final = self.to_hand_frame(wrist_pos_final)

        self.cur_targets[:, :3] = wrist_pos_final + add[:,:3]

        # if self.progress_buf[0] > 100:
        #     obj_pos_start = self.object_init_state[:, :3].clone()
        #     obj_pos_start[:, 2] += 0.25
        #     obj_pos_start = self.to_hand_frame(obj_pos_start)
        #     self.cur_targets[:,:3] = obj_pos_start

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))


        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def transform_object_to_hand_frame(self, object_info):
        hand_frame = self.hand_start_states[:, 0:7].clone()
        hand_position = hand_frame[:, :3]
        hand_quaternion = hand_frame[:, 3:]

        object_position = object_info[:, :3]
        object_quaternion = object_info[:, 3:7]
        object_linear_velo = object_info[:, 7:10]
        object_angular_velo = object_info[:, 10:]

        # get the inverse of the quaternion
        hand_quaternion = torch.cat([-hand_quaternion[:, 0:3], hand_quaternion[:, [-1]]], dim=-1)
        R_hand = quaternion_to_matrix(hand_quaternion)
        #R_hand = R_hand.permute(0, 2, 1)
        # get transpose of rotation matrix
        transformed_position = (R_hand @ (object_position - hand_position).unsqueeze(-1)).squeeze(-1)
        transformed_quaternion = quaternion_multiply(hand_quaternion,object_quaternion)  # Update according to your quaternion multiplication rule
        transformed_linear_velo = (R_hand @ object_linear_velo.unsqueeze(-1)).squeeze(-1)
        transformed_angular_velo = (R_hand @ object_angular_velo.unsqueeze(-1)).squeeze(-1)

        return torch.cat([transformed_position, transformed_quaternion, transformed_linear_velo, transformed_angular_velo],
                         dim=-1)

    def to_hand_frame(self, input):
        root_pos = self.hand_start_states[:, :3].clone()
        root_rot = self.hand_start_states[:, 3:7].clone()
        input = input - root_pos
        inverse_rot = torch.cat([-root_rot[:, 0:3], root_rot[:, [-1]]], dim=-1)
        R_hand = quaternion_to_matrix(inverse_rot)
        output = (R_hand @ input.unsqueeze(-1)).squeeze(-1)
        return output


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        #self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.compute_observations()
        self.compute_reward(self.actions)



#####################################################################
###=========================jit functions=========================###
#####################################################################


#torch.jit.script
# def compute_hand_reward(
#     rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
#     max_episode_length: float, object_pos, object_rot,obj_linevel,obj_angvel, obj_init_state, hand_pos,
#     dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
#     actions, action_penalty_scale: float,
#     success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
#     fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool,finger_force, table_force
# ):
#     reward_dict = {}
#     action_penalty = torch.sum(actions ** 2, dim=-1)
#     obj_vel_penalty = torch.sum(obj_linevel ** 2, dim=-1) + torch.sum(obj_angvel ** 2, dim=-1)
#
#     resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
#
#     impulse = torch.norm(finger_force[...,:3],dim=-1).sum(-1) - torch.norm(table_force[...,:3],dim=-1).sum(-1)# + 50*9.81
#
#     obj_height = object_pos[...,2] - 0.225
#     height_reward = torch.clamp(obj_height, min=0,max=0.5)
#
#     # obj_movement = torch.norm(object_pos-obj_init_state[...,:3],dim=-1)
#     # object_rot[:] = obj_init_state[...,3:7] = 0.5
#     # quat_diff = quat_mul(object_rot, quat_conjugate(obj_init_state[...,3:7]))
#     # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
#
#     reward = action_penalty * action_penalty_scale + obj_vel_penalty*-0.5 + height_reward*0 + impulse*5
#     reward_dict['action_penalty'] = action_penalty.mean().item()* action_penalty_scale
#     reward_dict['obj_vel_penalty'] = obj_vel_penalty.mean().item()*-0.5
#     reward_dict['height_reward'] = height_reward.mean().item()*0
#     reward_dict['impulse'] = impulse.mean().item()*5
#     return reward, resets, reward_dict, progress_buf, successes, 0




@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


import os
import numpy as np
import torch
import torch.nn.functional as F
from icecream import ic
from pylotier.utils.timer import Timer
from pylotier.utils import main_utils
import torchvision
import pylotier.utils.flow_viz as flow_viz
from src.deepLearning.FlowFormer.core.FlowFormer import build_flowformer
from src.deepLearning.FlowFormer.configs.submission import get_cfg
from src.deepLearning.FlowFormer.core.utils.flow_viz import  flow_to_image
import src.modules.foot_columns.column_helpers as ch 
import src.modules.foot_columns.column_kernels as ck 

from icecream import ic
from pylotier.utils.depth_visualizer_torch import visualize_depth
from pylotier.utils.flow_viz import tau_to_vis
from src.deepLearning.FlowFormer.configs.submission import get_cfg
from scipy.spatial.transform import Rotation as R
import cv2

class CreSF():
       
    def __init__(self, 
                 model_path, 
                 n_iter, 
                 height, 
                 width, 
                 fx, 
                 fy,
                 cx,
                 cy,
                 baseline,
                 device="cuda",
                 frame_distance=2):

        ### initialize constants
        # self.prev_xyz = None
        # self.prev_left = None
        self.fx = fx
        self.fy = fy 
        self.cx = cx 
        self.cy = cy
        self.h = height
        self.w = width
        self.device = device
        self.n_iter = n_iter
        self.baseline = baseline

        cfg = get_cfg()
        self.model = torch.nn.DataParallel(build_flowformer(cfg))

        #### load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        #### send model to the GPU
        self.model.to(self.device)
        self.model.eval()

        ### store the past x number of frames, so we can inference on a larger frame distance
        self.past_rgb_frames = []
        self.past_xyz_frames = []
        self.past_transf_mtxs = []
        # self.past_depth_frames = []
        self.frame_history_length = frame_distance + 1

    ### takes in numpy, outputs as numpy
    def infer_images(self, img1, img2):
        # print("Model Forwarding...")
        imgL = img1.transpose(2, 0, 1)
        imgR = img2.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])

        imgL = torch.tensor(imgL.astype("float32")).to(self.device)
        imgR = torch.tensor(imgR.astype("float32")).to(self.device)

        imgL_dw2 = F.interpolate(
            imgL,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        imgR_dw2 = F.interpolate(
            imgR,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        # print(imgR_dw2.shape)
        with torch.inference_mode():
            # pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

            # prediction = self.model(imgL, imgR, iters=self.n_iter, flow_init=None)
            prediction = self.model(imgL, imgR)


        flow_former_out = prediction[0].squeeze(0).permute(1,2,0).cpu().numpy()
        flow_former_out = cv2.resize(flow_former_out, (self.w, self.h))

        return flow_former_out


    def infer_depth(self, left_img, right_img):
        pred_disp = np.abs(self.infer_images(left_img, right_img))
        # ic(pred_disp)
        pred_disp = pred_disp[:, :, 0]

        depth = (self.fx * self.baseline) / pred_disp

        depth[depth > 100.0] = 100
        depth[np.isnan(depth)] = 0.01
        depth[depth < 0.01] = 0.01
        depth[np.isinf(depth)] = 100
        depth[np.isposinf(depth)] = 100
        depth[np.isneginf(depth)] = 100

        return depth, pred_disp
    
    def infer_flow_self(self, left_prev, left_current):
        pred_flow = self.infer_images(left_current, left_prev)

        return pred_flow


    # def infer_flow(self, left):
    #     ### first frame condition
    #     if self.prev_left is None:
    #         self.prev_left = left.copy()
    #         flow = np.zeros((self.h, self.w, 2), dtype=np.float32)
    #         return flow

    #     else:
    #         pred_flow = self.infer_images(self.prev_left, left)
    #         self.prev_left = left.copy()
    #         return pred_flow


    #Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
    def inference(self, left, right, transf_mtx):

        # rel_rot_vec = np.asarray(
        #     [-rel_rot_vec[0], -rel_rot_vec[1], rel_rot_vec[2]]
        # )
        # rel_transl_vec = np.asarray(
        #     [rel_transl_vec[0], rel_transl_vec[1], rel_transl_vec[2]]
        # )

        ### first frame condition
        if len(self.past_rgb_frames) < self.frame_history_length:
            depth, disparity = self.infer_depth(left, right)
            curr_xyz = self.compute_xyz_from_depth(depth)

            self.past_rgb_frames.append(left)
            self.past_xyz_frames.append(curr_xyz)
            self.past_transf_mtxs.append(transf_mtx)

            sceneflow = np.zeros_like(left, dtype=np.float32)
            flow = np.zeros((self.h, self.w, 2), dtype=np.float32) + 5.0
            disparity = np.zeros((self.h, self.w), dtype=np.float32)
            depth = np.zeros((self.h, self.w), dtype=np.float32)
            vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)

            return sceneflow, vis, flow, vis, disparity, depth, vis, flow, vis, flow, vis,
                    
        else:
            ### update rgb and transf memory 
            self.past_transf_mtxs.append(transf_mtx)
            self.past_transf_mtxs.pop(0)
            self.past_rgb_frames.append(left)
            self.past_rgb_frames.pop(0)
            
            ### and get the combined transf mtx
            acc_transf_mtx = main_utils.get_combined_transf_mtxs(self.past_transf_mtxs)
            rotaion_mtx = acc_transf_mtx[:3, :3]
            rotation_vec = R.from_matrix(rotaion_mtx).as_rotvec()
            translation_vec = acc_transf_mtx[:3, 3]

            ### compute flow and depth
            flow = self.infer_flow_self(self.past_rgb_frames[0], self.past_rgb_frames[-1])
            depth, disparity = self.infer_depth(left, right)
            curr_xyz = self.compute_xyz_from_depth(depth)

            ### update point cloud memory
            self.past_xyz_frames.append(curr_xyz)
            self.past_xyz_frames.pop(0)

            ### compute index image 
            linspace_w = np.linspace(0, int(self.w) - 1, int(self.w))
            linspace_h = np.linspace(0, int(self.h) - 1, int(self.h))
            grids = np.meshgrid(linspace_w, linspace_h)
            # ic(grids)
            grid_mesh = np.dstack((grids[0], grids[1]))

            # map = np.zeros_like(flow)
            # map[index_flow[:,:,:1], index_flow[:,:,:0]] = grid_mesh
            # ic(grid_mesh.shape)

            index_flow = (grid_mesh + flow).astype(np.int32)
            index_flow[:,:,0] = np.clip(index_flow[:,:,0], 0, self.w - 2.0)
            index_flow[:,:,1] = np.clip(index_flow[:,:,1], 0, self.h - 2.0)
            # ic(index_flow.shape)
            # ic(self.prev_xyz.shape)
            # ic(curr_xyz.shape)

            ### compute dynamic flow
            # transformation_mtx = main_utils.get_total_transformation_torch(self.past_transf_mtxs[0], curr_trannsf_mtx)
            dynamic_flow, induced_flow = self.compute_dynamic_flow(flow, depth, rotation_vec, translation_vec)
            dynamic_flow_mag = np.sqrt(np.power(dynamic_flow[:, :, 0], 2) + np.power(dynamic_flow[:, :, 1], 2))
            ########################

            ### compute sceneflow
            sceneflow =  curr_xyz[index_flow[:,:,1], index_flow[:,:,0]] - self.past_xyz_frames[-1]
            sceneflow[disparity < 1.0] = [0.0, 0.0, 0.0]
            sceneflow[dynamic_flow_mag < 3.0] = [0.0, 0.0, 0.0]

            ### visualize
            ###########################
            # flow_vis = flow_to_image(flow)
            flow_vis = flow_viz.flow_to_image_clip(flow)
            depth_vis = visualize_depth(torch.from_numpy(depth).to(self.device), max_depth=100).cpu().numpy()
            sceneflow_vis = tau_to_vis(sceneflow, min_max=10.0)
            # ic(left.dtype)
            # ic(depth_vis.dtype)
            # ic(flow_vis.dtype)
            # ic(sceneflow_vis.dtype)

            induced_flow_vis = flow_viz.flow_to_image_clip(induced_flow)
            dynamic_flow_vis = flow_viz.flow_to_image_clip(dynamic_flow)


            # combined_vis = cv2.vconcat([cv2.hconcat([left, depth_vis.astype(np.uint8)]), cv2.hconcat([flow_vis, sceneflow_vis])])
            # combined_rgb = cv2.vconcat([left, right])
            # cv2.imshow("combined_vis", combined_vis)
            # cv2.imshow("combined_rgb", combined_rgb)
            # cv2.waitKey(0)
            ###########################

            # self.prev_left = left
            # self.prev_xyz = curr_xyz



            # ic(np.max(dynamic_flow))

            return (sceneflow, 
                    sceneflow_vis, 
                    flow, 
                    flow_vis,
                    disparity, 
                    depth, 
                    depth_vis,
                    induced_flow,
                    induced_flow_vis,
                    dynamic_flow,
                    dynamic_flow_vis
                    )



    def compute_xyz_from_depth(self, depth):
        linspace_w = np.linspace(0, int(self.w) - 1, int(self.w))
        linspace_h = np.linspace(0, int(self.h) - 1, int(self.h))
        grid = np.meshgrid(linspace_w, linspace_h)
        U = grid[0]
        V = grid[1]

        # ic(U.shape)
        # ic(V.shape)
        # ic(depth.shape)

        
        Z = depth.copy()
        X = Z * (U - self.cx) / self.fx
        Y = Z * (V - self.cy) / self.fy

        xyz = np.dstack([X, Y, Z])
        return xyz 


    def compute_dynamic_flow(self, np_flow, depth, rel_rot_vec, rel_transl_vec):
        ### Computing dynamic flow
        np_induced_flow = ck.run_generate_induced_flow_img(depth,
                                                                self.h,
                                                                self.w,
                                                                self.fx,
                                                                self.fy,
                                                                self.cx,
                                                                self.cy,
                                                                rel_rot_vec,
                                                                rel_transl_vec)
        
        # if np.max(np_flow) > 0.1:
        #     np_induced_flow_vis = flow_viz.flow_to_image_clip(np_induced_flow)
        # else:
        # np_induced_flow_vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # np_dynamic_flow_vis = flow_viz.flow_to_image_clip(np_dynamic_flow)
        np_dynamic_flow = np_flow - np_induced_flow
        # ic(np_flow.shape)

        return np_dynamic_flow, np_induced_flow
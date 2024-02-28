import os
import numpy as np
import torch
import torch.nn.functional as F
from icecream import ic

from src.deepLearning.FlowFormer.core.FlowFormer import build_flowformer
from src.deepLearning.FlowFormer.configs.submission import get_cfg
from src.deepLearning.FlowFormer.core.utils.flow_viz import  flow_to_image
from src.deepLearning.FlowFormer.configs.submission import get_cfg
import src.modules.foot_columns.column_helpers as ch 
import src.modules.foot_columns.column_kernels as ck 

from pylotier.utils.timer import Timer  
from pylotier.utils import main_utils
import pylotier.utils.flow_viz as flow_viz
from pylotier.utils.depth_visualizer_torch import visualize_depth
from pylotier.utils.flow_viz import tau_to_vis
from scipy.spatial.transform import Rotation as R
import cv2

# from src.deepLearning.FlowFormer import sceneflow as SceneFlowModule

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
                 fps,
                 device="cuda",
                 frame_distance=3):

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
        self.original_fps = fps

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


    #Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
    def infer_opticalflow(self, curr_left, curr_right, curr_transf_mtx, intrinsics, cre_depth = None, cre_disparity=None, of=None):

        # rel_rot_vec = np.asarray(
        #     [-rel_rot_vec[0], -rel_rot_vec[1], rel_rot_vec[2]]
        # )
        # rel_transl_vec = np.asarray(
        #     [rel_transl_vec[0], rel_transl_vec[1], rel_transl_vec[2]]
        # )

        ### first frame condition
        if len(self.past_rgb_frames) < self.frame_history_length:
            if cre_depth is not None:
                depth, disparity = cre_depth, cre_disparity
            else:
                depth, disparity = self.infer_depth(curr_left, curr_right)
            curr_xyz = self.compute_xyz_from_depth(depth, intrinsics)

            self.past_rgb_frames.append(curr_left)
            self.past_xyz_frames.append(curr_xyz)
            self.past_transf_mtxs.append(curr_transf_mtx)

            sceneflow = np.zeros_like(curr_left, dtype=np.float32)
            flow = np.zeros((self.h, self.w, 2), dtype=np.float32) + 5.0
            disparity = np.zeros((self.h, self.w), dtype=np.float32)
            depth = np.zeros((self.h, self.w), dtype=np.float32)
            vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)

            return None, None, None, None
                    
        else:
            ### update rgb and transf mtx memory 
            self.past_transf_mtxs.append(curr_transf_mtx)
            self.past_transf_mtxs.pop(0)
            self.past_rgb_frames.append(curr_left)
            self.past_rgb_frames.pop(0)
            

            ### compute flow and depth
            if of is not None:
                flow = of
            else:   
                # ic(len(self.past_rgb_frames))
                flow =  - self.infer_flow_self(self.past_rgb_frames[0], self.past_rgb_frames[-1]) 
                # cv2.imshow("0", self.past_rgb_frames[0])
                # cv2.imshow("-1", self.past_rgb_frames[-1])

            # flow = flow * -1
            if cre_depth is not None:
                depth, disparity = cre_depth, cre_disparity
            else:
                depth, disparity = self.infer_depth(curr_left, curr_right)


            ### get the combined transf mtx
            combined_transf_mtx = main_utils.get_combined_transf_mtxs(self.past_transf_mtxs, forward=False)
            # combined_transf_mtx = self.past_transf_mtxs[-1]


            prev_flow_pcd_xyz = self.past_xyz_frames[0]

            return (depth, flow, prev_flow_pcd_xyz, combined_transf_mtx)


    def compute_xyz_from_depth(self, depth, intrinsics):
        linspace_w = np.linspace(0, int(intrinsics["width"]) - 1, int(intrinsics["width"]))
        linspace_h = np.linspace(0, int(intrinsics["height"]) - 1, int(intrinsics["height"]))
        grid = np.meshgrid(linspace_w, linspace_h)
        U = grid[0]
        V = grid[1]

        # ic(U.shape)
        # ic(V.shape)
        # ic(depth.shape)

        
        Z = depth.copy()
        X = Z * (U - intrinsics["cx"]) / intrinsics["fx"]
        Y = Z * (V - intrinsics["cy"]) / intrinsics["fy"]

        xyz = np.dstack([X, Y, Z])
        return xyz 
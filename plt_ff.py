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




# ███████╗██╗      ██████╗ ██╗    ██╗    ███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗
# ██╔════╝██║     ██╔═══██╗██║    ██║    ████╗ ████║██╔═══██╗██╔══██╗██║   ██║██║     ██╔════╝
# █████╗  ██║     ██║   ██║██║ █╗ ██║    ██╔████╔██║██║   ██║██║  ██║██║   ██║██║     █████╗  
# ██╔══╝  ██║     ██║   ██║██║███╗██║    ██║╚██╔╝██║██║   ██║██║  ██║██║   ██║██║     ██╔══╝  
# ██║     ███████╗╚██████╔╝╚███╔███╔╝    ██║ ╚═╝ ██║╚██████╔╝██████╔╝╚██████╔╝███████╗███████╗
# ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝     ╚═╝     ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝
                                                                                            

class FlowFormer():

    def __init__(flow_module, config=None, device = None, show_flag=False):

        flow_module.config = config
        flow_module.model_path = flow_module.config['flowformer']['model_path'] 
        flow_module.device = device
        flow_module.skip = int(config['flowformer']['skip'])

        # ic(flow_module.skip)
        flow_module.frames = []
        flow_module.just_last_frame = None
        flow_module.transformation_matrix_frames = []
        flow_module.show_flag = show_flag

        torch.backends.cudnn.benchmark = True

        flow_module.focal_length = fx
        flow_module.baseline = baseline

        flow_module.max_depth = max_depth

        #### initialize model
        cfg = get_cfg()
        flow_module.model = torch.nn.DataParallel(build_flowformer(cfg))

        #### load weights
        checkpoint = torch.load(flow_module.model_path, map_location=flow_module.device)
        flow_module.model.load_state_dict(checkpoint)
        
        #### send model to the GPU
        flow_module.model.to(flow_module.device)
        flow_module.model.eval()

    # ╔═╗╦═╗╔═╗╔═╗╦═╗╔═╗╔═╗╔═╗╔═╗╔═╗  ╦╔╗╔╔═╗╦ ╦╔╦╗
    # ╠═╝╠╦╝║╣ ╠═╝╠╦╝║ ║║  ║╣ ╚═╗╚═╗  ║║║║╠═╝║ ║ ║ 
    # ╩  ╩╚═╚═╝╩  ╩╚═╚═╝╚═╝╚═╝╚═╝╚═╝  ╩╝╚╝╩  ╚═╝ ╩ 
    
    def preprocess_input(flow_module, img1, img2):

        inference_size = (384, 672)
        # else:
        #     print("Wrong Input shape")

        image1 = img1[:,:,:3].permute(2, 0, 1).float()
        image2 = img2[:,:,:3].permute(2, 0, 1).float()

        image1 = image1.unsqueeze(0).to(flow_module.device)
        image2 = image2.unsqueeze(0).to(flow_module.device)

        image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)

        return image1, image2
    
    # ╔═╗╔═╗╔═╗╔╦╗╔═╗╦═╗╔═╗╔═╗╔═╗╔═╗╔═╗  ╔═╗╦ ╦╔╦╗╔═╗╦ ╦╔╦╗
    # ╠═╝║ ║╚═╗ ║ ╠═╝╠╦╝║ ║║  ║╣ ╚═╗╚═╗  ║ ║║ ║ ║ ╠═╝║ ║ ║ 
    # ╩  ╚═╝╚═╝ ╩ ╩  ╩╚═╚═╝╚═╝╚═╝╚═╝╚═╝  ╚═╝╚═╝ ╩ ╩  ╚═╝ ╩ 
    
    def postprocess_output(flow_module, flow_prediction):

        inference_size = (384, 672)
        # else:
        #     print("Wrong Input shape")
        
        original_size = (376, 672)

        #### resizing flow back to original image size
        flow_pr = F.interpolate(flow_prediction, size=original_size, mode='bilinear', align_corners=True)
        #### scale down the flow values themselves as they are scaled because of the image resize
        flow_pr[:, 0] = flow_pr[:, 0] * original_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * original_size[-2] / inference_size[-2]

        #### remove batch dimension
        flow_out = flow_pr.squeeze(0)  ## (H, W, 2) dimensions

        return flow_out

    # ╔╦╗╔═╗╦╔╗╔  ╦╔╗╔╔═╗╔═╗╦═╗
    # ║║║╠═╣║║║║  ║║║║╠╣ ║╣ ╠╦╝
    # ╩ ╩╩ ╩╩╝╚╝  ╩╝╚╝╚  ╚═╝╩╚═

    @torch.no_grad()
    def infer(flow_module, torch_img1, transformation_matrix, oclr_enable = False):


        with Timer(name="Flowformer Torch\ - 376 x 672,creating cuda stream"):
            stream1 = torch.cuda.Stream(flow_module.device)
        #### save the original size because GMflow needs to use a multiple of 32, so we need to resize it 

    
        with torch.cuda.stream(stream1):

                with Timer(name="Flowformer Torch\ - 376 x 672,pre-processing data - convert to torch"):
                    flow_module.original_image_size = (torch_img1.shape[0], torch_img1.shape[1])

                if len(flow_module.frames) > flow_module.skip:
                    image1, image2 = flow_module.preprocess_input(flow_module.frames[-1], flow_module.frames[0])

                    total_relative_transformation = main_utils.get_total_transformation_torch(flow_module.transformation_matrix_frames[0],
                                                                                            flow_module.transformation_matrix_frames[-1])
                    
                    flow_module.frames.pop(0)
                    flow_module.frames.append(torch_img1.clone())
                    flow_module.transformation_matrix_frames.pop(0)
                    flow_module.transformation_matrix_frames.append(transformation_matrix.copy())

                    with Timer(name="Flowformer Torch\ - 376 x 672,Flowformer Torch - 4090 - 376 x 672"):
                        fwd_output = flow_module.model(image1, image2,
                                                attn_splits_list=[2],
                                                corr_radius_list=[-1],
                                                prop_radius_list=[-1])

                    with Timer(name="Flowformer Torch\ - 376 x 672,post-processing data"):
                        flow = fwd_output['flow_preds'][-1]
                        flow = flow_module.postprocess_output(flow)

                    # if flow_module.show_flag:  
                    #     with Timer(name="Flowformer Torch\ - 376 x 672,flow-uv to flow-rgb"):
                    #         flow_rgb = torchvision.utils.flow_to_image(flow)
                    #         flow_rgb = flow_rgb.permute(1,2,0) ### (H, W, 3)
                    # else:
                    flow_rgb = torch.zeros_like(torch_img1)
                    # with Timer(name="Flowformer Torch\ - 376 x 672,OCLR-bwd check"):
                    ### for OCLR stuff
                    if oclr_enable:
                        bwd_output = flow_module.model(image2, image1,
                                                attn_splits_list=[2],
                                                corr_radius_list=[-1],
                                                prop_radius_list=[-1])
                        
                        flow_bwd = bwd_output['flow_preds'][-1]
                        flow_bwd = flow_module.postprocess_output(flow_bwd)
                        
                        flow_rgb_bwd = torchvision.utils.flow_to_image(flow_bwd)
                        flow_rgb_bwd = flow_rgb_bwd.permute(1,2,0) ### (H, W, 3)
                    else:
                        flow_bwd = None
                        flow_rgb_bwd = None

                else:
                    flow_rgb = torch.zeros_like(torch_img1)
                    flow = torch.zeros( (2, torch_img1.shape[0], torch_img1.shape[1]))

                    ### for OCLR
                    flow_bwd = torch.zeros( (2, torch_img1.shape[0], torch_img1.shape[1]))
                    flow_rgb_bwd = torch.zeros_like(torch_img1)
                    total_relative_transformation = torch.eye(4)

                    flow_module.frames.append(torch_img1.clone())
                    flow_module.transformation_matrix_frames.append(transformation_matrix.copy())
                    return flow, flow_rgb, total_relative_transformation, flow_bwd, flow_rgb_bwd

        torch.cuda.synchronize()

        
        return flow, flow_rgb, total_relative_transformation, flow_bwd, flow_rgb_bwd
       

    @torch.no_grad()
    def just_infer(flow_module, torch_img1, HARD_CODED_SCALE, horizontal_only = False):


        # with Timer(name="Flowformer Torch\ - 376 x 672,creating cuda stream"):
        #     stream1 = torch.cuda.Stream(flow_module.device)
        #### save the original size because GMflow needs to use a multiple of 32, so we need to resize it 

    
        # with torch.cuda.stream(stream1):

        with Timer(name="Flowformer Torch\ - 376 x 672,pre-processing data - convert to torch"):
            flow_module.original_image_size = (torch_img1.shape[0], torch_img1.shape[1])

        if flow_module.just_last_frame is not None:

            image1, image2 = flow_module.preprocess_input(torch_img1.clone(), flow_module.just_last_frame)

            flow_module.just_last_frame = torch_img1.clone()


            with Timer(name="Flowformer Torch\ - 376 x 672,Flowformer Torch - 4090 - 376 x 672"):
                fwd_output = flow_module.model(image1, image2)
                ic(fwd_output)

            with Timer(name="Flowformer Torch\ - 376 x 672,post-processing data"):
                flow = fwd_output[0]
                flow = flow_module.postprocess_output(flow)


        else:
            flow = torch.zeros( (2, torch_img1.shape[0], torch_img1.shape[1]))

            flow_module.just_last_frame = torch_img1.clone()

        # torch.cuda.synchronize()

        np_gmflow = flow.permute(1, 2, 0).cpu().numpy()
        if horizontal_only:
            np_gmflow[:,:,1] *= 0

        np_flow_vis, flow_scale = flow_viz.flow_to_image_scale(np_gmflow, HARD_CODED_SCALE)

        
        return np_gmflow, np_flow_vis, flow_scale
    


    @torch.no_grad()
    def infer_disp(flow_module, left, right, HARD_CODED_SCALE, horizontal_only = False):


        # with Timer(name="Flowformer Torch\ - 376 x 672,creating cuda stream"):
        #     stream1 = torch.cuda.Stream(flow_module.device)
        #### save the original size because GMflow needs to use a multiple of 32, so we need to resize it 

    
        # with torch.cuda.stream(stream1):

        with Timer(name="Flowformer depth Torch\ - 376 x 672,pre-processing data - convert to torch"):
            flow_module.original_image_size = (left.shape[0], left.shape[1])

            image1, image2 = flow_module.preprocess_input(left.clone(), right.clone())

            # with Timer(name="Flowformer Torch\ - 376 x 672,Flowformer Torch - 4090 - 376 x 672"):
            fwd_output = flow_module.model(image1, image2)
                # ic(fwd_output)

            # with Timer(name="Flowformer Torch\ - 376 x 672,post-processing data"):
            flow = fwd_output[0]
            ic(flow.shape)


    def infer_depth(self, left, right):

        disp = self.infer(left, right)
        # ic(disp.dtype)

        # with Timer("DEPTH,post-processing"):
        
        depth = self.focal_length*self.baseline/disp
       
        depth[depth > self.max_depth] = self.max_depth
        depth[torch.isnan(depth)] = self.max_depth
        depth[torch.isinf(depth)] = self.max_depth

        return depth, disp
            


import torch
import numpy as np

from os import path as osp
import shutil
from PIL import Image
import pickle as pkl
import scipy
import sys
import os
import cv2

sys.path.append('./third_party/human_body_prior')

import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.transform import resize

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV
)
from contraints import JOINT_RANGE

import pytorch3d
import pytorch3d.transforms

def glo_rot_withVpose(angle_temp = np.array([0.0, 0.0, 0.0])):
    if angle_temp.shape[0] == 3: 
        ang_pare = torch.from_numpy(angle_temp)
        Rx = pytorch3d.transforms.axis_angle_to_matrix(torch.from_numpy(np.array([np.pi,0.0,0.0]))) 
        ang_cam_frame = pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(torch.mm(pytorch3d.transforms.axis_angle_to_matrix(ang_pare),Rx))
    elif angle_temp.shape[1] == 3: 
        ang_pare = torch.from_numpy(angle_temp)
        Rx = pytorch3d.transforms.axis_angle_to_matrix(torch.from_numpy(np.array([np.pi,0.0,0.0]))) 
        ang_cam_frame = pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(torch.matmul(pytorch3d.transforms.axis_angle_to_matrix(ang_pare),Rx))
    else:
        return np.array([0.0, 0.0, 0.0])
    return ang_cam_frame.numpy().copy()

def clean(x):
	if 'chumpy' in str(type(x)):
		return np.array(x)
	elif type(x) == scipy.sparse.csc.csc_matrix:
		return x.toarray()
	else:
		return x

def mask2bbox(mask):
    x_mask = mask.sum(0)
    y_mask = mask.sum(1)
    x = np.nonzero(x_mask)[0].min()
    w = np.nonzero(x_mask)[0].max() - x + 1
    y = np.nonzero(y_mask)[0].min()
    h = np.nonzero(y_mask)[0].max() - y + 1
    bbox_xywh = np.array([x,y,w,h,1.0])

    scale = 1.2
    x_center = int((np.nonzero(x_mask)[0].min()+np.nonzero(x_mask)[0].max()+1)/2)
    y_center = int((np.nonzero(y_mask)[0].min()+np.nonzero(y_mask)[0].max()+1)/2)
    x_large = x_center - scale*(w/2)
    y_large = y_center - scale*(h/2)

    bbox_xywh = np.array([x_large,y_large,scale*w,scale*h,1.0])
    return bbox_xywh

def pkl2npz(bm_fname):
	if '.pkl' in bm_fname:
		hack_bm_path = bm_fname[:-4]+'_lqh.npz'
		with open(bm_fname, "rb") as f:
			try:
				data = pkl.load(f, encoding="latin1")
			except ModuleNotFoundError as e:
				if "chumpy" in str(e):
					message = ("Failed to load pickle file because "
						"chumpy is not installed.\n"
						"The original SMPL body model archives store some arrays as chumpy arrays, these are cast back to numpy arrays before use but it is not possible to unpickle the data without chumpy installed.")
					raise ModuleNotFoundError(message) from e
				else:
					raise e
			data = {k: clean(v) for k,v in data.items() if type(v)}
			data = {k: v for k,v in data.items() if type(v) == np.ndarray}
			np.savez(hack_bm_path, **data)

	else:
		hack_bm_path = bm_fname
	return hack_bm_path

def rotate_bound_black_bg(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH))

def padding_occluder(image, size=1080, center=[0,0]):

    (h, w) = image.shape[:2]
    (hX, hY) = (h // 2, w // 2)
    (cX, cY) = center
    (cX, cY) = (int(cX), int(cY))

    new_img = np.zeros((size,size,3))
    if (h, w)!=(hX*2, hY*2):
        new_img[cX-hX:cX+hX+1,cY-hY:cY+hY+1,:] = image
    else:
        new_img[cX-hX:cX+hX,cY-hY:cY+hY,:] = image
 
    return new_img


class PoseRender():
    def __init__(self, output_path='./data_generated/generated_images', texture_id=-100, num_poses=50, proc_id=-1, withOcc=False):
        super(PoseRender, self).__init__()

        self.num_poses = num_poses
        if proc_id==-1:
            self.output_path = output_path
        else:
            self.output_path = './exps/Proc_'+str(proc_id)+'/data_generated/generated_images'
        self.proc_id = proc_id
        self.withOcc = withOcc

        # Setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        #################################
        ########### step 1: using vposer to get pose  —— vpose part
        self.support_dir = './utils_PoseExaminer/support_data/dowloads'
        self.expr_dir = osp.join(self.support_dir,'vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
        self.bm_fname =  osp.join(self.support_dir,'models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0_lqh.npz')
        self.sample_amass_fname = osp.join(self.support_dir, 'amass_sample.npz')# a sample npz file from AMASS

        ####### Loading SMPLx Body Model ####### 
        from human_body_prior.body_model.body_model import BodyModel
        self.bm_fname = pkl2npz(self.bm_fname)
        self.bm = BodyModel(bm_fname=self.bm_fname).to(self.device)

        ####### Loading VPoser Body Pose Prior ####### 
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        self.vp, self.ps = load_model(self.expr_dir, model_code=VPoser,
                                      remove_words_in_model_weights='vp_model.',
                                      disable_grad=True)
        self.vp = self.vp.to(self.device)

        #################################
        ########### step 2: add uv map —— pytorch3d render part

        self.support_dir_uv = './utils_PoseExaminer/support_data/uv_bk_map'
        self.data_filename = os.path.join(self.support_dir_uv, "UV_Processed.mat")
        self.tex_filename = os.path.join(self.support_dir_uv,"texture_from_SURREAL.png")
        self.verts_filename = os.path.join(self.support_dir_uv, "basicmodel_m_lbs_10_207_0_v1.0.0.pkl")

        with open( os.path.join(self.support_dir_uv, 'UV_all', 'male_train.txt' ) ) as f:
            self.txt_paths = f.read().splitlines() 
        self.cloth_img_name = self.txt_paths[texture_id]
        self.tex_filename = os.path.join(self.support_dir_uv,self.cloth_img_name)

        self.bk_path = os.path.join(self.support_dir_uv,'bk_all','bk_pool','bk_park.jpg')
        with Image.open(self.bk_path) as image:
            self.bk_image = np.asarray(image.convert("RGB")).astype(np.float32)


        ###### Load texture data
        self.ALP_UV = loadmat(self.data_filename)
        with Image.open(self.tex_filename) as image:
            self.np_image = np.asarray(image.convert("RGB")).astype(np.float32)
        self.tex = torch.from_numpy(self.np_image / 255.)[None].to(self.device)

        self.verts = torch.from_numpy((self.ALP_UV["All_vertices"]).astype(int)).squeeze().to(self.device) # (7829,)
        self.U = torch.Tensor(self.ALP_UV['All_U_norm']).to(self.device) # (7829, 1)
        self.V = torch.Tensor(self.ALP_UV['All_V_norm']).to(self.device) # (7829, 1)
        self.faces = torch.from_numpy((self.ALP_UV['All_Faces'] - 1).astype(int)).to(self.device)  # (13774, 3)
        self.face_indices = torch.Tensor(self.ALP_UV['All_FaceIndices']).squeeze()  # (13774,)

        ###### Load occluder:
        if self.withOcc:
            raise NotImplementedError("You need to provide the occluder patch, then can use this")
            self.occder_filename = './utils_PoseExaminer/data/occluder/occluder_bag.jpg'
            with Image.open(self.occder_filename) as image:
                self.occ_image = np.asarray(image.convert("RGB")).astype(np.float32)


        ###### Map each face to a (u, v) offset
        self.offset_per_part = {}
        self.already_offset = set()
        self.cols, self.rows = 4, 6
        for i, u in enumerate(np.linspace(0, 1, self.cols, endpoint=False)):
            for j, v in enumerate(np.linspace(0, 1, self.rows, endpoint=False)):
                self.part = self.rows * i + j + 1  # parts are 1-indexed in face_indices
                self.offset_per_part[self.part] = (u, v)

        self.U_norm = self.U.clone()
        self.V_norm = self.V.clone()

        # iterate over faces and offset the corresponding vertex u and v values
        for i in range(len(self.faces)):
            self.face_vert_idxs = self.faces[i]
            self.part = self.face_indices[i]
            self.offset_u, self.offset_v = self.offset_per_part[int(self.part.item())]
            
            for vert_idx in self.face_vert_idxs:   
                # vertices are reused, but we don't want to offset multiple times
                if vert_idx.item() not in self.already_offset:
                    # offset u value
                    self.U_norm[vert_idx] = self.U[vert_idx] / self.cols + self.offset_u
                    # offset v value
                    # this also flips each part locally, as each part is upside down
                    self.V_norm[vert_idx] = (1 - self.V[vert_idx]) / self.rows + self.offset_v
                    # add vertex to our set tracking offsetted vertices
                    self.already_offset.add(vert_idx.item())

        # invert V values
        self.V_norm = 1 - self.V_norm

        ########### Initialize a camera.
        # World coordinates +Y up, +X left and +Z in.
        self.R, self.T = look_at_view_transform(90, 0, 0) 
        self.T[0,1] += 0.4 
        self.cameras = PerspectiveCameras(focal_length=60,device=self.device, R=self.R, T=self.T)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. 
        self.raster_settings = RasterizationSettings(
            image_size=1080, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Place a point light in front of the person. 
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights
            )
        )

    def pose_inverse(self, pose):
        distribution_zgen = self.vp.encode(pose).mean
        threshold_zgen = (distribution_zgen.abs()>2).sum(1)
        return threshold_zgen

    def render(self, Zgen, low=None, high=None, pose_mean=None, pose_std=None, occ_position=None, with_bkground=False, phase=1, sample_dim=None, boundary=None, step_gap=0.02, random_glo=False):

        consider_glo = False
        consider_shape = False
        if Zgen.shape[1]>32: 
            if Zgen.shape[1]==35:
                consider_glo = True
                glo_zgen = Zgen[:,:3]
                glo_zgen = glo_zgen * torch.tensor((0.1,1.0,0.1)).to(glo_zgen.device)
                Zgen = Zgen[:,3:]
            elif Zgen.shape[1]==42:
                consider_shape = True
                shape_zgen = Zgen[:,-10:]
                Zgen = Zgen[:,:-10]
            else:
                pass
        else:
            pass
        
        if phase==1 or phase==4:
            num_poses = 1
        elif phase==2:
            num_poses = self.num_poses
        elif phase==3 or phase==5:
            num_poses = self.num_poses
        
        if boundary is not None:
            num_poses = self.num_poses
        
        joint_range = torch.from_numpy(np.array(JOINT_RANGE).T * np.pi).to(Zgen.device)

        multi_sampled_pose_body_mean = self.vp.decode(Zgen)['pose_body'].contiguous().view(Zgen.shape[0], -1).detach()

        threshold_zgen = None
        constraint_zgen = None

        if phase==2:
            if low is not None:
                if sample_dim is not None:           
                    multi_sampled_pose_body = torch.zeros((50,63)).to(high.device)
                    border_log_probs = torch.zeros((50,63)).to(high.device)
                    sample_dim = torch.from_numpy(sample_dim).to(high.device)
                    top = multi_sampled_pose_body_mean+high*((sample_dim[0]-sample_dim[1])!=-1)
                    bottom = multi_sampled_pose_body_mean-low*((sample_dim[1]-sample_dim[0])!=-1)
                    numDim = (sample_dim==1).sum().item()
                    constraint_zgen = sample_dim[0]+sample_dim[1]
                    if numDim!= 5:
                        raise NotImplementedError('Not released yet : Rrender_pose1')

                    sign,dim = torch.where(sample_dim==1)
                    for k in range(numDim):
                        bottom_temp = bottom.clone()
                        top_temp = top.clone()
                        if sign[k]==0:
                            bottom_temp[0,dim[k]] = top[0,dim[k]]
                            top_temp[0,dim[k]] = top[0,dim[k]] + step_gap
                            dist = torch.distributions.Uniform(bottom_temp, top_temp)
                            sampled_pose_temp = dist.sample(torch.Size((10,))).squeeze()
                            multi_sampled_pose_body[k*10:(k+1)*10] = sampled_pose_temp
                            border_log_probs[k*10:(k+1)*10] = dist.log_prob(sampled_pose_temp)
                            while border_log_probs[k*10:(k+1)*10].isinf().sum() != 0:
                                sampled_pose_temp = dist.sample(torch.Size((10,))).squeeze()
                                multi_sampled_pose_body[k*10:(k+1)*10] = sampled_pose_temp
                                border_log_probs[k*10:(k+1)*10] = dist.log_prob(sampled_pose_temp)
                        elif sign[k]==1:
                            top_temp[0,dim[k]] = bottom[0,dim[k]]
                            bottom_temp[0,dim[k]] = bottom[0,dim[k]] - step_gap
                            dist = torch.distributions.Uniform(bottom_temp, top_temp)
                            sampled_pose_temp = dist.sample(torch.Size((10,))).squeeze()
                            multi_sampled_pose_body[k*10:(k+1)*10] = sampled_pose_temp
                            border_log_probs[k*10:(k+1)*10] = dist.log_prob(sampled_pose_temp)
                            while border_log_probs[k*10:(k+1)*10].isinf().sum() != 0:
                                sampled_pose_temp = dist.sample(torch.Size((10,))).squeeze()
                                multi_sampled_pose_body[k*10:(k+1)*10] = sampled_pose_temp
                                border_log_probs[k*10:(k+1)*10] = dist.log_prob(sampled_pose_temp)
                        threshold_zgen = self.pose_inverse(sampled_pose_temp)
                        if (threshold_zgen>5).sum()>1 :
                            constraint_zgen[dim[k]] = 0.
                else:
                    top = multi_sampled_pose_body_mean+high
                    bottom = multi_sampled_pose_body_mean-low
                    dist = torch.distributions.Uniform(bottom, top)
                    multi_sampled_pose_body = dist.sample(torch.Size((num_poses,))).squeeze()
                               
                    border_log_probs = dist.log_prob(multi_sampled_pose_body)
                    while border_log_probs.isinf().sum() != 0:
                        multi_sampled_pose_body = dist.sample(torch.Size((num_poses,))).squeeze()     
                        border_log_probs = dist.log_prob(multi_sampled_pose_body)

                if border_log_probs.isnan().sum() != 0: 
                    raise NotImplementedError("Not released yet: Rrender_pose2")

            elif pose_mean is not None:
                mu = multi_sampled_pose_body_mean + pose_mean
                sig = pose_std
                dist = torch.distributions.Normal(mu, sig)
                multi_sampled_pose_body = dist.sample(torch.Size((num_poses,))).squeeze()
                border_log_probs = dist.log_prob(multi_sampled_pose_body)
            else:
                raise NotImplementedError("Not released yet : Rrender_pose3")

        elif boundary is not None:
            top = torch.min(multi_sampled_pose_body_mean+boundary,joint_range[0])
            bottom = torch.max(multi_sampled_pose_body_mean-boundary,joint_range[1])
            dist = torch.distributions.Uniform(bottom, top)
            multi_sampled_pose_body = dist.sample(torch.Size((num_poses,))).squeeze().to(torch.float32)     
            border_log_probs = None
        elif phase==3: # evaluation
            top = multi_sampled_pose_body_mean+high
            bottom = multi_sampled_pose_body_mean-low
            dist = torch.distributions.Uniform(bottom, top)
            multi_sampled_pose_body = dist.sample(torch.Size((num_poses,))).squeeze()
            border_log_probs = dist.log_prob(multi_sampled_pose_body)
        elif phase==5: # generation for adv training
            top = torch.min(multi_sampled_pose_body_mean+high,joint_range[0])
            bottom = torch.max(multi_sampled_pose_body_mean-low,joint_range[1])
            dist = torch.distributions.Uniform(bottom, top)
            multi_sampled_pose_body = dist.sample(torch.Size((num_poses,))).squeeze().to(torch.float32)   
            border_log_probs = dist.log_prob(multi_sampled_pose_body)
        else:
            border_log_probs = None
            multi_sampled_pose_body = multi_sampled_pose_body_mean.clone()
            top = None
            bottom = None


        if self.proc_id == -1:
            np.save('./data_generated/vposer_results',multi_sampled_pose_body.cpu().numpy())
        else:
            np.save('./exps/Proc_'+str(self.proc_id)+'/data_generated/vposer_results',multi_sampled_pose_body.cpu().numpy())

        # get vertices
        if consider_glo:
            rotation_angle = glo_zgen[0].to(torch.float64).cpu().numpy()
        elif random_glo:
            rotation_angle_range = torch.tensor((np.pi*0.05, 0.0, 0.0))  # AT-rV10g3lr05Rotdown (camera down, looking up)
            rotation_angle = rotation_angle_range * torch.rand((num_poses,3)) - torch.tensor((0.0,0.0,0.0)) #torch.tensor((0.0,np.pi/20,np.pi/20))
            rotation_angle = rotation_angle.to(torch.float64).numpy()
        else:
            rotation_angle=np.array([0.0, 0.0, 0.0])
        
        if consider_shape:
            body_shape_beta = shape_zgen.repeat(num_poses,1)
        else:
            body_shape=np.zeros(10) # average
            body_shape_beta = torch.from_numpy(body_shape[np.newaxis,:].repeat(num_poses,axis=0)).to(multi_sampled_pose_body.device).float()

        if random_glo:
            root_orient = torch.from_numpy(0.0-rotation_angle).to(multi_sampled_pose_body.device).float()
        else:
            root_orient = torch.from_numpy(0.0-rotation_angle[np.newaxis,:].repeat(num_poses,axis=0)).to(multi_sampled_pose_body.device).float()

        smpl_output = self.bm.forward(root_orient=root_orient, pose_body=multi_sampled_pose_body,betas=body_shape_beta,return_dict=True)

        ############## create our verts_uv values
        verts_uv = torch.cat([self.U_norm[None],self.V_norm[None]], dim=2) # (1, 7829, 2)

        out_img_dir = self.output_path
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        else:
            shutil.rmtree(out_img_dir)
            os.makedirs(out_img_dir)


        v_template = smpl_output['v'].to(self.device)
        bbox = []
        for idx in range(v_template.shape[0]):
            v_template_extended = v_template[idx][self.verts-1][None] # (1, 7829, 3)
            texture = TexturesUV(maps=self.tex, faces_uvs=self.faces[None], verts_uvs=verts_uv)
            mesh = Meshes(v_template_extended, self.faces[None], texture)

            images = self.renderer(mesh)
            output_img = images[0, ..., :3].cpu().numpy()
            output_img *= 255.0

            # compute mask and bounding box
            mask = images[0, ..., -1].cpu().numpy()
            mask[mask>0]=1
            mask_fore = mask[:,:,np.newaxis].copy()
            mask_back = 1 - mask_fore
            bbox.append(mask2bbox(mask_fore[:,:,0]))
            im_mask = Image.fromarray((mask*255.0).astype(np.uint8))
            if not os.path.exists(out_img_dir+'_mask'):
                os.makedirs(out_img_dir+'_mask')
            im_mask.save(out_img_dir+'_mask/image_%.5d.jpg'%idx)

            # background
            if with_bkground:
                if phase == 5:
                    with open( os.path.join(self.support_dir_uv, 'bk_all', 'bk_all.txt' ) ) as f:
                        bkimg_paths = f.read().splitlines() 
                    bkimg_len = len(bkimg_paths)
                    import random
                    bk_id = random.randint(0,bkimg_len-1)
                    bkimg_filename = os.path.join(self.support_dir_uv,bkimg_paths[bk_id])
                    with Image.open(bkimg_filename) as image:
                        if image.size!=(1080,1080):
                            image = image.resize((1080,1080))
                        self.bk_image = np.asarray(image.convert("RGB")).astype(np.float32)

                output_img = output_img * mask_fore + self.bk_image * mask_back
            
            if self.withOcc:
                if phase==4 or phase==5:
                    if phase==5: # add occluder randomly if you do not provide learned occlusion parameters
                        occ_position_sample = torch.tensor((480.0,480.,360.0)) * torch.rand(3) + torch.tensor((300.0,300.0,0))
                        occ_resize = np.random.random(2) + 0.6
                        occ_log_probs = None
                    else:
                        dist_occ = torch.distributions.Normal(occ_position, 0.05)
                        occ_position_sample = dist_occ.sample()
                        occ_log_probs = dist_occ.log_prob(occ_position_sample)
                        occ_resize = np.ones(2)

                    occ_image_rotated = rotate_bound_black_bg(self.occ_image, occ_position_sample[2].item()/2.0)
                    occ_image_rotated = resize(occ_image_rotated,np.insert((occ_image_rotated.shape[:2]*occ_resize)//2*2,2,3).astype(int))
                    occ_image_padded = padding_occluder(occ_image_rotated, size=output_img.shape[0], center=[occ_position_sample[0].item(),occ_position_sample[1].item()])
                    mask = occ_image_padded.sum(axis=-1)
                    mask[mask<=5.0]=0.0
                    mask[mask>5.0]=1.0
                    mask_occluder = mask[:,:,np.newaxis].copy()
                    mask_ori = 1 - mask_occluder
                    output_img = output_img * mask_ori + occ_image_padded * mask_occluder
                else:
                    occ_log_probs = None

            im = Image.fromarray(output_img.astype(np.uint8))
            im.save(out_img_dir+'/image_%.5d.jpg'%idx)
        
        if consider_shape:
            self.correct_annotation_mmhuman(rotation_angle,num_poses,bbox=bbox,shape=body_shape_beta.cpu().numpy())
        else:
            self.correct_annotation_mmhuman(rotation_angle,num_poses,bbox=bbox)
        
        if low is not None:
            if self.withOcc:
                return multi_sampled_pose_body_mean, root_orient, border_log_probs, top, bottom, constraint_zgen, occ_log_probs
            else:
                return multi_sampled_pose_body_mean, root_orient, border_log_probs, top, bottom, constraint_zgen
        else:
            return multi_sampled_pose_body_mean, root_orient, border_log_probs

    
    def correct_annotation_mmhuman(self,rotation_angle=np.array([0.0, 0.0, 0.0]),num_poses=50,bbox=None,shape=None,random=False):

        ori_name = './data/preprocessed_datasets/spin_pw3d_test.npz'

        if num_poses==50:
            ref_name = './utils_PoseExaminer/data_generated/generated_mmpose_init_50.npz'
        elif num_poses==200:
            ref_name = './utils_PoseExaminer/data_generated/generated_mmpose_init_200.npz'
        elif num_poses==500:
            ref_name = './utils_PoseExaminer/data_generated/generated_mmpose_init_500.npz'
        elif num_poses==1000: # CVPR submission
            ref_name = './utils_PoseExaminer/data_generated/generated_mmpose_init_1000.npz' 
        else:
            ref_name = './utils_PoseExaminer/data_generated/generated_mmpose_init.npz' 

        if random:
            out_name = './exps/Proc_'+str(self.proc_id)+'/data_generated_random/generated_mmpose.npz'
            vposer_name = './exps/Proc_'+str(self.proc_id)+'/data_generated_random/vposer_results.npy'
        else:
            out_name = './exps/Proc_'+str(self.proc_id)+'/data_generated/generated_mmpose.npz'
            vposer_name = './exps/Proc_'+str(self.proc_id)+'/data_generated/vposer_results.npy'

        ori_label = np.load(ori_name,allow_pickle=True)
        ref_label = np.load(ref_name)
        vpose = np.load(vposer_name)

        length_generated = vpose.shape[0]

        __key_strict__ = ori_label['__key_strict__']
        __temporal_len__ = np.array(length_generated)
        __keypoints_compressed__ = ori_label['__keypoints_compressed__']
        bbox_xywh = np.array(bbox)
        config = ori_label['config']
        smpl = ori_label['smpl']
        meta = ori_label['meta']

        image_path=[]
        for i in range(length_generated):
            if random:
                image_path.append('Proc_'+str(self.proc_id)+'/data_generated_random/'+ref_label['imgname'][i])
            else:
                image_path.append('Proc_'+str(self.proc_id)+'/data_generated/'+ref_label['imgname'][i])
        image_path = np.array(image_path)


        smpl.item()['body_pose'] = smpl.item()['body_pose'][:length_generated]
        for i in range(length_generated):
            smpl.item()['body_pose'][i][:21] = vpose[i].reshape(-1,3)

        smpl.item()['global_orient'] = smpl.item()['global_orient'][:length_generated]
        pose = ref_label['pose'][:length_generated]
        if np.abs(rotation_angle).sum()!=0:
            pose[:,0:3] = glo_rot_withVpose(rotation_angle) # from pare : 1-my_vposer_testset.py
        smpl.item()['global_orient'] = pose[:,0:3]

        smpl.item()['betas'] = smpl.item()['betas'][:length_generated]
        if shape is None:
            smpl.item()['betas'] = ref_label['shape'][:length_generated]
        else:
            smpl.item()['betas'] = shape

        meta.item()['gender'] = meta.item()['gender'][:length_generated]
        meta.item()['gender'] = ref_label['gender'][:length_generated]

        # TODO: Most of the code above are useless for the final mmhuman-based version.. Clean them up.

        np.savez(out_name, __key_strict__=__key_strict__, __temporal_len__=__temporal_len__, __keypoints_compressed__=__keypoints_compressed__,
            image_path=image_path, bbox_xywh=bbox_xywh, config=config, smpl=smpl, meta=meta)
        
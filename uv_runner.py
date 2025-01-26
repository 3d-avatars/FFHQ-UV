import time

import logging
import numpy as np
import os
import torch
from PIL import Image
from model import ours_fit_model_cropface630resize1024
from utils.visual_utils import Logger
from utils.data_utils import setup_seed

from RGB_Fitting.dataset.fit_dataset import FitDataset
from RGB_Fitting.utils.data_utils import tensor2np, img3channel, draw_mask, draw_landmarks, save_img
from FLAME_Apply_HIFI3D_UV.run_flame_apply_hifi3d_uv import read_mesh_obj, write_mesh_obj

logger = logging.getLogger(__name__)


class UvRunner:

    def __init__(self):
        setup_seed(123)

        self.device = "cuda"
        self.checkpoints_dir = "../checkpoints"
        self.topo_assets_dir = "../topo_assets"
        self.texgan_model_name = "texgan_ffhq_uv.pth"
        self.fit_dataset = FitDataset(
            lm_detector_path=os.path.join(self.checkpoints_dir, "lm_model/68lm_detector.pb"),
            mtcnn_detector_path=os.path.join(self.checkpoints_dir, "mtcnn_model/mtcnn_model.pb"),
            parsing_model_pth=os.path.join(self.checkpoints_dir, "parsing_model/79999_iter.pth"),
            parsing_resnet18_path=os.path.join(self.checkpoints_dir, "resnet_model/resnet18-5c106cde.pth"),
            lm68_3d_path=os.path.join(self.topo_assets_dir, "similarity_Lm3D_all.mat"),
            batch_size=1,
            device=self.device,
        )
        self.fit_model = ours_fit_model_cropface630resize1024.FitModel(
            cpk_dir=self.checkpoints_dir,
            topo_dir=self.topo_assets_dir,
            texgan_model_name=self.texgan_model_name,
            device=self.device
        )

    def generate_uv(
        self,
        input_file_path: str,
        output_dir_path: str,
    ) -> str:
        ffhq_uv_logger = Logger(
            vis_dir=output_dir_path,
            flag=f'texgan_{self.texgan_model_name[:-4]}',
            is_tb=True,
        )

        logger.info("Starting processing input image")

        tic = time.time()
        input_data = self.fit_dataset.get_input_data(input_file_path)
        
        basename = input_file_path[input_file_path.rfind("/") + 1: input_file_path.rfind(".")]

        torch.save(input_data, os.path.join(output_dir_path, f"{basename}_ffhq_uv.pt"))

        input_img = tensor2np(input_data["img"][:1, :, :, :])

        skin_img = tensor2np(input_data["skin_mask"][:1, :, :, :])
        skin_img = img3channel(skin_img)

        parse_mask = tensor2np(input_data["parse_mask"][:1, :, :, :], dst_range=1.0)
        parse_img = draw_mask(input_img, parse_mask)

        gt_lm = input_data["lm"][0, :, :].detach().cpu().numpy()
        gt_lm[..., 1] = input_img.shape[0] - 1 - gt_lm[..., 1]

        mask_image = Image.fromarray((img3channel(parse_mask) * 255).astype(np.uint8))
        orig_image = Image.fromarray((img3channel(input_img)).astype(np.uint8))

        mask_image.save(os.path.join(output_dir_path, "ffhq_uv_mask.png"))
        orig_image.save(os.path.join(output_dir_path, "orig.png"))

        lm_img = draw_landmarks(input_img, gt_lm, color="b")
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img], axis=1)
        save_img(combine_img, os.path.join(output_dir_path, f"{basename}_ffqh_uv_vis.png"))

        toc = time.time()
        logger.info(f"Finished processing input image: {input_file_path}, took {toc - tic:.4f} seconds.")

        logger.info(f"Starting fitting uv texture")
        if "trans_params" in input_data:
            input_data.pop("trans_params")

        input_data = {k: v.to(self.device) for (k, v) in input_data.items()}
        tic = time.time()
        self.fit_model.fitting(input_data=input_data, logger=ffhq_uv_logger)
        toc = time.time()

        logger.info(f"Finished fitting uv texture, took {toc - tic:.4f} seconds")
        return f"{output_dir_path}/stage3_uv.png"


    def apply_uv(
        self,
        input_mesh_path: str,
        output_mesh_path: str,
    ) -> (str, str):
        logger.info(f"Starting applying UV map for {input_mesh_path}")
        refer_mesh_path = './FLAME_Apply_HIFI3D_UV/flame2hifi3d_assets/FLAME_w_HIFI3D_UV.obj'
        # save_mesh_path = f'{input_mesh_path[:-4]}_w_HIFI3D_UV.obj'

        refer_data = read_mesh_obj(refer_mesh_path)
        flame_data = read_mesh_obj(input_mesh_path)

        flame_data['vt'] = refer_data['vt']
        flame_data['fvt'] = refer_data['fvt']

        write_mesh_obj(flame_data, output_mesh_path)

        eyes_vertices = []
        eyes_texture_vertices = []
        eyes_faces = []
        eyes_faces_textures = []

        old_vertex_index_to_new = {}
        texture_coords_to_new_index = {}

        for i in range(len(flame_data['v'])):
            if 3931 <= i <= 5022:
                eyes_vertices.append(flame_data['v'][i])
                old_vertex_index_to_new[i] = len(eyes_vertices) - 1

        for i in range(len(flame_data['fv'])):
            if 3931 <= flame_data['fv'][i][0] <= 5022:
                eyes_faces.append(
                    [
                        old_vertex_index_to_new[flame_data['fv'][i][0]],
                        old_vertex_index_to_new[flame_data['fv'][i][1]],
                        old_vertex_index_to_new[flame_data['fv'][i][2]],
                    ]
                )

                texture_indices = flame_data['fvt'][i]
                new_texture_indices = []
                for i in texture_indices:
                    texture_coords = (flame_data['vt'][i][0] - 2, flame_data['vt'][i][1])

                    if texture_coords not in texture_coords_to_new_index.keys():
                        eyes_texture_vertices.append(texture_coords)
                        texture_coords_to_new_index[texture_coords] = len(eyes_texture_vertices) - 1

                    new_texture_indices.append(texture_coords_to_new_index[texture_coords])

                eyes_faces_textures.append(new_texture_indices)

        eyes_data = {
            'v': np.array(eyes_vertices),
            'vt': np.array(eyes_texture_vertices),
            'fv': np.array(eyes_faces),
            'fvt': np.array(eyes_faces_textures)
        }

        eyeballs_mesh_file = f'{output_mesh_path[:-4]}_eyeballs.obj'

        write_mesh_obj(eyes_data, eyeballs_mesh_file)

        vertices_to_delete = []
        faces_to_delete = []

        for i in range(len(flame_data['fv'])):
            is_eye_face = True
            for j in range(3):
                if 3931 <= flame_data['fv'][i][j] <= 5022:
                    vertices_to_delete.append(flame_data['fv'][i][j])
                else:
                    is_eye_face = False

            if is_eye_face:
                faces_to_delete.append(i)

        flame_data['v'] = np.delete(flame_data['v'], vertices_to_delete, axis=0)
        flame_data['fv'] = np.delete(flame_data['fv'], faces_to_delete, axis=0)
        flame_data['fvt'] = np.delete(flame_data['fvt'], faces_to_delete, axis=0)

        head_mesh_file = f'{output_mesh_path[:-4]}_without_eyeballs.obj'

        write_mesh_obj(flame_data, head_mesh_file)
        logger.info(f"Finished applying UV map for {input_mesh_path}")

        return (eyeballs_mesh_file, head_mesh_file)

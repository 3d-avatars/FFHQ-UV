import time

import logging
import numpy as np
import os
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional

from .FLAME_Apply_HIFI3D_UV.run_flame_apply_hifi3d_uv import read_mesh_obj, write_mesh_obj
from .RGB_Fitting.dataset.fit_dataset import FitDataset
from .RGB_Fitting.model import ours_fit_model_cropface630resize1024
from .RGB_Fitting.utils.data_utils import setup_seed
from .RGB_Fitting.utils.data_utils import tensor2np, img3channel, draw_mask, draw_landmarks, save_img
from .RGB_Fitting.utils.visual_utils import Logger

logger = logging.getLogger(__name__)
file_path = str(Path(__file__).absolute())
dir_path = file_path[: file_path.rfind("/")]

class UvRunner:

    def __init__(self):
        setup_seed(123)

        self.device = "cuda"
        self.checkpoints_dir = f"{dir_path}/checkpoints"
        self.topo_assets_dir = f"{dir_path}/topo_assets"
        self.texgan_model_name = "texgan_cropface630resize1024_ffhq_uv_interpolate.pth"
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
        original_image_file_path: str,
        ffhq_uv_mask_path: str,
    ) -> str:
        logger.info("[UV Runner] Starting processing input image")

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

        mask_image.save(ffhq_uv_mask_path)
        orig_image.save(original_image_file_path)

        lm_img = draw_landmarks(input_img, gt_lm, color="b")
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img], axis=1)
        save_img(combine_img, os.path.join(output_dir_path, f"{basename}_ffqh_uv_vis.png"))

        toc = time.time()
        logger.info(f"[UV Runner] Finished processing input image: {input_file_path}, took {toc - tic:.4f} seconds.")

        logger.info(f"[UV Runner] Starting fitting uv texture")
        if "trans_params" in input_data:
            input_data.pop("trans_params")

        ffhq_uv_logger = Logger(
            vis_dir=output_dir_path,
            flag=f"texgan_{self.texgan_model_name[:-4]}",
            is_tb=True,
        )

        input_data = {k: v.to(self.device) for (k, v) in input_data.items()}
        tic = time.time()
        self.fit_model.fitting(input_data=input_data, logger=ffhq_uv_logger)
        toc = time.time()

        logger.info(f"[UV Runner] Finished fitting uv texture, took {toc - tic:.4f} seconds")
        return f"{output_dir_path}/stage3_uv.png"

    def apply_uv(
        self,
        input_mesh_path: str,
        output_mesh_path: str,
        should_save_eyes_separately: True
    ) -> (str, Optional[str], Optional[str]):
        logger.info(f"[UV Runner] Starting applying UV map for {input_mesh_path}")
        refer_mesh_path = f"{dir_path}/FLAME_Apply_HIFI3D_UV/flame2hifi3d_assets/FLAME_w_HIFI3D_UV.obj"

        refer_data = read_mesh_obj(refer_mesh_path)
        head_data = read_mesh_obj(input_mesh_path)

        head_data["vt"] = refer_data["vt"]
        head_data["fvt"] = refer_data["fvt"]
        head_data.pop("mtl_name", None)

        write_mesh_obj(head_data, output_mesh_path)

        (eyes_data, half_eyes_data) = self.__get_eyes_mesh(head_data)
        head_data = self.__remove_eyes_from_head(head_data)

        if should_save_eyes_separately:
            eyeballs_full_mesh_file = f"{output_mesh_path[:-4]}_eyes_full.obj"
            eyeballs_half_mesh_file = f"{output_mesh_path[:-4]}_eyes_half.obj"

            write_mesh_obj(eyes_data, eyeballs_full_mesh_file)
            write_mesh_obj(half_eyes_data, eyeballs_half_mesh_file)
        else:
            eyeballs_full_mesh_file = None
            eyeballs_half_mesh_file = None

        head_mesh_file = output_mesh_path
        write_mesh_obj(head_data, head_mesh_file)

        logger.info(f"[UV Runner] Finished applying UV map for {input_mesh_path}")

        return (head_mesh_file, eyeballs_full_mesh_file, eyeballs_half_mesh_file)

    def __get_eyes_mesh(
        self,
        head_data: Dict[str, np.ndarray],
    ) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        logger.info(f"[UV Runner] Starting retrieving eyes mesh from head")

        eyes_vertices = []
        eyes_texture_vertices = []
        eyes_faces = []
        eyes_faces_textures = []

        old_vertex_index_to_new = {}
        texture_coords_to_new_index = {}

        for i in range(len(head_data["v"])):
            if 3931 <= i <= 5022:
                eyes_vertices.append(head_data["v"][i])
                old_vertex_index_to_new[i] = len(eyes_vertices) - 1

        for i in range(len(head_data["fv"])):
            if 3931 <= head_data["fv"][i][0] <= 5022:
                eyes_faces.append(
                    [
                        old_vertex_index_to_new[old_index]
                        for old_index
                        in head_data["fv"][i]
                    ]
                )

                texture_indices = head_data["fvt"][i]
                new_texture_indices = []
                for i in texture_indices:
                    texture_coords = (head_data["vt"][i][0] - 2, head_data["vt"][i][1])

                    if texture_coords not in texture_coords_to_new_index.keys():
                        eyes_texture_vertices.append(texture_coords)
                        texture_coords_to_new_index[texture_coords] = len(eyes_texture_vertices) - 1

                    new_texture_indices.append(
                        texture_coords_to_new_index[texture_coords]
                    )

                eyes_faces_textures.append(new_texture_indices)

        eyes_data = {
            "v": np.array(eyes_vertices),
            "vt": np.array(eyes_texture_vertices),
            "fv": np.array(eyes_faces),
            "fvt": np.array(eyes_faces_textures),
        }
        half_eyes_data = self.__remove_half_of_eyes(eyes_data)
        logger.info(f"[UV Runner] Finished retrieving eyes mesh from head")

        return eyes_data, half_eyes_data

    def __remove_half_of_eyes(
        self,
        eyes_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        logger.info(f"[UV Runner] Starting removing half of eyes")

        vertices = eyes_data["v"]
        faces = eyes_data["fv"]
        faces_textures = eyes_data["fvt"]

        faces_count_half = int(faces.shape[0] / 2)

        # vertices from which we will
        anchor_vertex_indices = [961, 415]

        # find all faces that contain anchor vertices
        adjacent_faces_indices = np.where(
            np.isin(faces, anchor_vertex_indices).any(axis=1)
        )[0]

        faces_indices_to_remove = set(adjacent_faces_indices)
        polygons_to_remove_count = len(faces_indices_to_remove)

        while polygons_to_remove_count < faces_count_half:
            # find all faces adjacent to current
            new_adjacent_indices = np.where(
                np.isin(faces, faces.take(list(faces_indices_to_remove), axis=0)).any(axis=1)
            )[0]

            old_len = len(faces_indices_to_remove)

            faces_indices_to_remove = faces_indices_to_remove | set(new_adjacent_indices.data)
            polygons_to_remove_count += len(faces_indices_to_remove) - old_len

        # remove faces and textures
        remaining_faces = np.delete(faces, list(faces_indices_to_remove), axis=0)
        remaining_face_textures = np.delete(faces_textures, list(faces_indices_to_remove), axis=0)

        # remove unused vertices
        used_vertices_mask = np.zeros(len(vertices), dtype=bool)
        used_vertices_mask[np.unique(remaining_faces)] = True

        remaining_vertices = vertices[used_vertices_mask]

        # update vertices mapping for faces
        remapping = np.zeros(len(vertices), dtype=int) - 1
        remapping[np.where(used_vertices_mask)[0]] = np.arange(len(remaining_vertices))

        updated_faces = remapping[remaining_faces]

        eyes_data = {
            "v": remaining_vertices,
            "vt": eyes_data["vt"],
            "fv": updated_faces,
            "fvt": remaining_face_textures,
        }

        logger.info(f"[UV Runner] Finished removing half of eyes")
        return eyes_data

    def __remove_eyes_from_head(
        self,
        head_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        logger.info(f"[UV Runner] Starting removing eyes from head mesh")

        vertices_to_delete = []
        faces_to_delete = []

        for i in range(len(head_data["fv"])):
            is_eye_face = True
            for j in range(3):
                if 3931 <= head_data["fv"][i][j] <= 5022:
                    vertices_to_delete.append(head_data["fv"][i][j])
                else:
                    is_eye_face = False

            if is_eye_face:
                faces_to_delete.append(i)

        head_data["v"] = np.delete(head_data["v"], vertices_to_delete, axis=0)
        head_data["fv"] = np.delete(head_data["fv"], faces_to_delete, axis=0)
        head_data["fvt"] = np.delete(head_data["fvt"], faces_to_delete, axis=0)

        logger.info(f"[UV Runner] Finished removing eyes from head mesh")

        return head_data

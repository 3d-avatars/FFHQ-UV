#!/bin/bash
set -e


######################### Configuration #########################
# mesh_path: the path of the mesh of the head which is without eyeballs.
#################################################################
mesh_path=../input_mesh/pixar-egor_w_HIFI3D_UV.obj


#################### Run Add Eyeball Script #####################
# Read the head mesh of ${mesh_path}
# Save the eyeball mesh add texture in the directory of ${mesh_path}
#################################################################
cd ./Mesh_Add_EyeBall
python run_mesh_add_eyeball.py --mesh_path ${mesh_path}

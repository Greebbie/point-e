# Point·E


This work based on [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://arxiv.org/abs/2212.08751).

# Usage

Extend from Point-E paper

To Start
Install with `pip install -e .`.

We clean up our code in mainly the following notebooks:
* [simple_train.ipynb](point_e/examples/simple_train.ipynb) - the notebook contains 2 sections preload the data and training & load the model to virtualize
* [sepe.py](point_e/models/sepe.py) - the file contains our main model architecture extended from PointDiffusionTransformer where fuse the text and img inputs
* [fusion.py](point_e/models/fusion.py) - the file contains TextImgFusion Class where crossattention is used

To get started with examples, see the following notebooks:

 * [image2pointcloud.ipynb](point_e/examples/image2pointcloud.ipynb) - sample a point cloud, conditioned on some example synthetic view images.
 * [text2pointcloud.ipynb](point_e/examples/text2pointcloud.ipynb) - use our small, worse quality pure text-to-3D model to produce 3D point clouds directly from text descriptions. This model's capabilities are limited, but it does understand some simple categories and colors.
 * [pointcloud2mesh.ipynb](point_e/examples/pointcloud2mesh.ipynb) - try our SDF regression model for producing meshes from point clouds.


# Dataset

Bundle of a 2D reference img, 3D point cloud and text caption for diffusion.

Cap3D [here](https://huggingface.co/datasets/tiange/Cap3D)

uses the ABO version which captioned with BLIP [here](https://huggingface.co/docs/transformers/en/model_doc/blip).

The script for process the downloaded data and generating the dataset refer to (* [download_data.py](point-e/download_data.py) and * [generate_dataset.py](point-e/generate_dataset.py)). The generated dataset.jsonl file need to be placed at upper level folder for the uid match.

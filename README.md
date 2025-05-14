# Semantic Enhanced Point-E

This work based on [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://arxiv.org/abs/2212.08751).

The purpose of Semantic Enhanced Point-E is to improve the quality and semantic accuracy of 3D point cloud generation from complex text prompts. Extended from original work mode between text prompt to 3D point cloud generation and image to 3D point cloud reconstruction. By incorporating advanced semantic enhancements, this project builds on the original Point-E system to produce 3D models that more precisely align with detailed and nuanced descriptions, enabling richer and more contextually accurate representations.

# Usage

Extend from Point-E paper. 

To Start
Install with `pip install -e .`.

The code notebooks are cleaned up in mainly the following notebooks:
* [simple_train.ipynb](point_e/examples/simple_train.ipynb) - the notebook contains 2 sections preload the data and training & load the model to virtualize
* [sepe.py](point_e/models/sepe.py) - the file contains our main model architecture extended from PointDiffusionTransformer where fuse the text and img inputs. We takes pre-trained Base 40M weights.
* [fusion.py](point_e/models/fusion.py) - the file contains TextImgFusion Class where crossattention is used

# Dataset

Bundle of a 2D reference img, 3D point cloud and text caption for diffusion.

Cap3D [here](https://huggingface.co/datasets/tiange/Cap3D)

uses the ABO version which captioned with BLIP [here](https://huggingface.co/docs/transformers/en/model_doc/blip).

The script for process the downloaded data and generating the dataset refer to (* [download_data.py](point-e/download_data.py) and * [generate_dataset.py](point-e/generate_dataset.py)). The generated dataset.jsonl file need to be placed at upper level folder for the uid match.

A larger size of training data with rich varities of types are strongly encouraged for better performance.

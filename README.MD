# Towards a size-aware implicit 2-D cloth rendering

Bastian Scharnagl, Christian Groth\
*7th IEEE International Conference on Artificial Intelligence & eXtended and Virtual Reality (AIxVR 2025)*

![3799_header.jpg](https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/3799_header.jpg)

Load the original VAE from [Huggingface](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.ckpt) and place it in checkpoints folder.\
Load our trained checkpoint from [Huggingface](https://huggingface.co/BastianScharnagl/size-aware-tryon) and place it in checkpoints folder.

With inference.ipynb you can process the images from the sample_images folder. You can change the prompt for customized results.\
For your own images, you have to preprocess it and get the Human Keypoints with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), the Fashion Keypoints with [KGDet](https://github.com/ShenhanQian/KGDet) and the cloth-agnostic-mask with [this script](https://github.com/bastianscharnagl/size-aware-tryon/tools/get_agnostic_mask.py).

Sample images with prompt "model posing in blue jeans trousers".

<div>
<p float="left">
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_cfg_scale_9.00_-30.jpg" width="100" />
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_cfg_scale_9.00_0.0.jpg" width="100" />
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_cfg_scale_9.00_30.jpg" width="100" />
</p>
</div>

Sample inpainting images.
<div>
<p float="left">
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_cfg_scale_9.00_-30.jpg" width="100" />
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_cfg_scale_9.00_0.0.jpg" width="100" />
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_cfg_scale_9.00_30.jpg" width="100" />
</p>
</div>

<div>
<p float="left">
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_inpainting_-30.jpg" width="100" />
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_inpainting_0.0.jpg" width="100" />
  <img src="https://github.com/BastianScharnagl/size-aware-tryon/blob/main/assets/013582_samples_inpainting_30.jpg" width="100" />
</p>
</div>


Based on the following research:

[ControlNet](https://github.com/lllyasviel/ControlNet)\
[Stable Diffusion](https://github.com/Stability-AI/stablediffusion/tree/main)\
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)\
[KGDet](https://github.com/ShenhanQian/KGDet)\
[Dress Code](https://github.com/aimagelab/dress-code)
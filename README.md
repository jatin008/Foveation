# Foveation
____________________________________________________________________________________________________

In order to provide an immersive visual experience, modern displays require head mounting, high image resolution, low latency, as well as high refresh rate. This poses a challenging computational problem. On the other hand, the human visual system can consume only a tiny fraction of this video stream due to the drastic acuity loss in the peripheral vision. Foveated rendering and compression can save computations by reducing the image quality in the peripheral vision. However, this can cause noticeable
artifacts in the periphery, or, if done conservatively, would provide only modest savings. In this project, we have developed FoveaGAN which employs the recent advances in generative adversarial neural networks. 

In the FoveaGAN, we reconstruct a plausible peripheral video from a small fraction of pixels provided in every frame. The reconstruction is done by finding the closest matching video to this sparse input stream of pixels on the learned manifold of different videos.

#### Project
____________________________________________________________________________________________________

FoveaGAN model for Foveation has been implemented on Pytorch <br/>

Prerequisites:
```
numpy==1.17.3
Pillow==6.2.1
opencv-python
```
Install requirements.txt available in the FoveaGAB folder using the following command
```
pip install -r requirements.txt
```

Steps to run the code are as follows:

1. Clone the github repo.

2. Install Python from:-
        https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe

3. Open command prompt and go to the folder location "FoveaGAN"
        cd "FoveaGAN"

4. If the  computer has Nvidia Graphic Card then
        pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

    else 
        pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

##### To train the network (Pretrained weights are already present, so, no need to train the network)
____________________________________________________________________________________________________

1. Open command prompt and go to the folder location "FoveaGAN"
        cd "FoveaGAN"
2. Type:
        python "train.py"
        
##### Use the pretrained network to generate Foveated Video
____________________________________________________________________________________________________
1. Put the test images in folder sampled_images

2. Open command prompt and go to folder location "GANS"
        cd "FoveaGAN"
        
3. Type:
        python "foveate_video.py"
        
4. Output will be in the folder "Output"

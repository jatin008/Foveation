1. Install Python from:-
        https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe
 
2. Open command prompt and go to the folder location "FoveaGAN"
        cd "FoveaGAN"

3. Open command prompt and enter
        pip install -r requirements.txt

4. If the  computer has Nvidia Graphic Card then
        pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

    else 
        pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


To train the network (Pretrained weights are already present, so, no need to train the network)
===================

1. Open command prompt and go to the folder location "FoveaGAN"
        cd "FoveaGAN"
2. Type:
        python "train.py"


Use the pretrained network to generate Foveated Video
=======================================================

1. Put the test images in folder sampled_images
2. Open command prompt and go to folder location "GANS"
        cd "FoveaGAN"
3. Type:
        python "foveate_video.py"
4. Output will be in the folder "Output"
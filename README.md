# Foveation

In order to provide an immersive visual experience, modern displays require head mounting, high image resolution, low latency, as well as high refresh rate. This poses a challenging computational problem. On the other hand, the human visual system can consume only a tiny fraction of this video stream due to the drastic acuity loss in the peripheral vision. Foveated rendering and compression can save computations by reducing the image quality in the peripheral vision. However, this can cause noticeable
artifacts in the periphery, or, if done conservatively, would provide only modest savings. In this project, we have developed FoveaGAN which employs the recent advances in generative adversarial neural networks. 

In the FoveaGAN, we reconstruct a plausible peripheral video from a small fraction of pixels provided in every frame. The reconstruction is done by finding the closest matching video to this sparse input stream of pixels on the learned manifold of different videos.

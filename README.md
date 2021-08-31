# Membrans
This is some scripts for preparing image data, augmentations, training neural networks - U-Net, VAE at now, and saving their results as images on disk.

Results:

Orig:

![alt text](./example/orig0250.png)

Mask:

![alt text](./example/bone0250.png)

Different colors results of VAE:

![alt text](./example/bone_membr0250.png)

One color results of VAE:

![alt text](./example/bone_membr_one_color0250.png)

Generated membran:

![alt text](./example/membr0250.png)

As you can see, result of VAE doesn't valuable. The main reason is low amount of data for learning for this project

Other architecture of second NN may give better results, so it can be improved
* VAEN1 - seed 2202, x - orig, y - full one color results - indicate teeth, make brighter when already known, make dimly when defect
* VAEN2 - seed 2202, x - orig, y - one color membran - full black
* Unet_VAE - seed 2202, x - NG, y - one color membran - full black, 2 epoch, batch size - 3
* GAN - seed 2202, x - ng, y one color membran, - don't learn decoder to do membran -> output is bone as usual, but generate bone on places, where it may be and mustn't be
* GAN - seed 2202, x - ng, y one color membran, -> some epoches have high acc (49,41,39,36,35,33,32,26,24,23,22,21,20,19,14,13,10,9,8,7) - focused on teeths and bone
* Det - seed 2202, x - g, y - r, 1/10 bone/membran -> can increase acc, but at test and train give zero or more spaces with wrong form, but localised, 1 epoch, scale - 8
* Det - seed 2202, x - g, y - r, 1~1 bone/membran -> strange middle line of points, and more times seen something like a membran, but in wrong places
* Det - seed 2202, x - g, y - r, 1~1 bone/membran, regularization -> strange middle line
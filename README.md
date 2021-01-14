## Super Resolution of occluded or unclear faces using Generative Adversarial Networks

### Authors
**Akhil Nair | Mounica Subramani | Mrinal Soni | SuRui Yang**

#### SUMMARY

Image or video data contain plenty of information and have a wide range of applications in the field of research and development. Many applications require zooming of a specific area of interest in the image where in high resolution becomes essential, e.g.  tumors diagnosis, visual surveillance and autonomous vehicle navigation.   However,  no matter how well an upscaling algorithm performs, there will still be some amount of information lost and drop in image quality.  An efficient fix for certain situation deserves high attention.

Estimating  a  high-resolution  (HR)  image  from  its  low-resolution  (LR)  counterpart  is  referred to  as  super-resolution  (SR).  In this project, we aim to explore image super-resolution using generative adversarial networks, trying to minimize detail lost during the upscaling procedure. 

We have built a model for generator and discriminator in the first phase of the project. The model is successfully giving out an image. We are looking forward to work on the second phase of the project. It includes optimizing the model and extend the implementation towards video footage super resolution.

#### PROPOSED  PLAN  OF  RESEARCH - PHASE 2

The timeline of the project is one full semester (September - December). A variation of GAN called SRGAN that employs a deep learning network inspired from ResNet and diverges from the traditional Mean Square Error (MSE) loss function to a perceptual loss function is used. We have done data processing and developed a SRGAN model in first phase. The second phase includes optimizing the model to bring the output image from generator, close to Super Resolution image which is as good as high resolution or better than high resolution image and scale it in cloud. Based  on  the  results  of  the  experiments  we  would  like  to  extend  our  methods  and  algorithms towards video footage Super Resolution.

First step of model optimization is migrating the implementation to cloud set up. It would avoid memory issues and helps us utilize the whole data set. Once implemented in the cloud platform, we would try further optimizations like, see if we can reduce the number of trainable parameters without considerably affecting the performance of the model, train on the entire dataset. Also, train for a higher number of iterations and try optimizing the loss functions.

The  implementation  of  the  model  is  done  using  tensorflow’s  TFGAN  library  and  trained using TPU’s available on google cloud platform. 

#### PRELIMINARY  RESULTS  AND  DATASETS

![Image of Yaktocat](https://github.ccs.neu.edu/mounicasubramani/DS5500-Project-SRGAN/blob/master/Images/Fig%201%20Downsampling.png)

The super resolution GAN models are experimented on combination of widely used benchmark dataset Celeb-A and on Indian Movie Face database (IMFDB). Celeb- A is a large-scale facial attributes dataset with 202,599 face images of 10,177 unique identities. The images are mostly frontal images and less occluded which might create a bias in the model. IMFDB is a large unconstrained face database consisting of 34512 images of 100 Indian actors collected from more than 100 videos. Unlike the Celeb-A dataset the faces in IMFDB are collected from videos collected from the last two decades by manual selection and cropping of video frames resulting in diversity in age, poses, dress patterns, expressions etc.

We used Python PIL library to downsample the images from 218x178 to 64x64, 32x32 i.e. high resolution and low resolution respectively. These are the low resolution images fed to the Generator model. Once fed into our model the generator takes the low resolution images as input and tries to identify the shape, colour and texture of objects in our images, generating a new fake image (called SR) based on what was learned from the LR images. These fake / generated images were then fed to the discriminator which also takes in the actual HR images as an input. Based on their quantitative values, discriminator classifies each image as fake (close to 0) or real (close to 1). For now, the discriminator model works fine in classifying between the SR images and the HR images. In the second phase, we are trying to improve generator training so that it performs better there by fooling discriminator. 

##### High Resolution Image | Low Resolution Images | Super Resolution Images
![](https://github.ccs.neu.edu/mounicasubramani/DS5500-Project-SRGAN/blob/master/Images/HR_Image.jpg)
![](https://github.ccs.neu.edu/mounicasubramani/DS5500-Project-SRGAN/blob/master/Images/LR_Image.jpg)
![](https://github.ccs.neu.edu/mounicasubramani/DS5500-Project-SRGAN/blob/master/Images/SR_Image.jpg)

##### Image obtained at 1st Iteration
![](https://github.ccs.neu.edu/mounicasubramani/DS5500-Project-SRGAN/blob/master/Images/0.jpeg)

##### Image obtained at 100th Iteration
![](https://github.ccs.neu.edu/mounicasubramani/DS5500-Project-SRGAN/blob/master/Images/100_cr.jpeg)

#### REFERENCES

[1]  C. Ledig,  L. Theis,  F. Husz ar,  J. Caballero,  A. Cunningham,A. Acosta,  A. Aitken,  A. Te- jani, J. Totz, Z. Wang, et al.Photo-realistic single image super-resolution using a gener-ative adversarial network.arXiv preprint arXiv:1609.04802,2016

[2]  A. Radford, L. Metz, and S. Chintala. Unsupervised repre-sentation learning with deep convo- lutional generative adver-sarial networks.arXiv preprint arXiv:1511.06434, 2015.

[3]  J. Kim, J. Kwon Lee, and K. Mu Lee. Accurate image super-resolution using very deep con- volutional networks.  InTheIEEE Conference on Computer  Vision and Pattern  Recogni-tion (CVPR), June 2016.

[4]  Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning faceattributes in the wild. InProceedings of International Con-ference on Computer Vision (ICCV), 2015.

[5]  S. Setty, M. Husain, P. Beham, J. Gudavalli, M. Kandasamy, R. Vaddi, V. Hemadri, J C Karure, R. Raju, Rajan, V. Kumar and C V Jawahar. Indian Movie Face Database:  A Benchmark for Face Recognition Under Wide Variations, National Conference on Computer Vision, Pattern Recognition, Image Processing and Graphics (NCVPRIPG), 2013

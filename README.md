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

#### REFERENCES

[1]  C. Ledig,  L. Theis,  F. Husz ar,  J. Caballero,  A. Cunningham,A. Acosta,  A. Aitken,  A. Te- jani, J. Totz, Z. Wang, et al.Photo-realistic single image super-resolution using a gener-ative adversarial network.arXiv preprint arXiv:1609.04802,2016

[2]  A. Radford, L. Metz, and S. Chintala. Unsupervised repre-sentation learning with deep convo- lutional generative adver-sarial networks.arXiv preprint arXiv:1511.06434, 2015.

[3]  J. Kim, J. Kwon Lee, and K. Mu Lee. Accurate image super-resolution using very deep con- volutional networks.  InTheIEEE Conference on Computer  Vision and Pattern  Recogni-tion (CVPR), June 2016.

[4]  Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning faceattributes in the wild. InProceedings of International Con-ference on Computer Vision (ICCV), 2015.

[5]  S. Setty, M. Husain, P. Beham, J. Gudavalli, M. Kandasamy, R. Vaddi, V. Hemadri, J C Karure, R. Raju, Rajan, V. Kumar and C V Jawahar. Indian Movie Face Database:  A Benchmark for Face Recognition Under Wide Variations, National Conference on Computer Vision, Pattern Recognition, Image Processing and Graphics (NCVPRIPG), 2013

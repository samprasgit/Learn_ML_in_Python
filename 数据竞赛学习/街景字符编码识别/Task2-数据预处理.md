## 2.1图像读取

- ### Pillow

  Pillow是Python图像处理函式库(PIL）的一个分支。Pillow提供了常见的图像读取和处理的操作，而且可以与ipython notebook无缝集成，是应用比较广泛的库。  

  Pillow的官方文档：https://pillow.readthedocs.io/en/stable/

- ### OpenCV

  OpenCV是一个跨平台的计算机视觉库，最早由Intel开源得来。OpenCV发展的非常早，拥有众多的计算机视觉、数字图像处理和机器视觉等功能。OpenCV在功能上比Pillow更加强大很多，学习成本也高很多。 

  OpenCV包含了众多的图像处理的功能，OpenCV包含了你能想得到的只要与图像相关的操作。此外OpenCV还内置了很多的图像特征处理算法，如关键点检测、边缘检测和直线检测等

  OpenCV官网：https://opencv.org/     

  OpenCV Github：https://github.com/opencv/opencv      

  OpenCV 扩展算法库：https://github.com/opencv/opencv_contrib

## 2.2图像数据扩增方法(Data Augementation)

在深度学习中数据扩增方法非常重要，数据扩增可以增加训练集的样本，同时也可以有效缓解模型过拟合的情况，也可以给模型带来的更强的泛化能力。

在深度学习中数据扩增方法非常重要，数据扩增可以增加训练集的样本，同时也可以有效缓解模型过拟合的情况，也可以给模型带来的更强的泛化能力。

#### 数据扩增方法：

数据扩增方法有很多：从颜色空间、尺度空间到样本空间，同时根据不同任务数据扩增都有相应的区别

- 图像分类，数据扩增一般不会改变标签
- 物体检测，数据扩增会改变物体坐标位置
- 图像分割，数据扩增会改变像素标签

在常见的数据扩增方法中，一般会从图像颜色、尺寸、形态、空间和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。         

以`torchvision`为例，常见的数据扩增方法包括：

- transforms.CenterCrop      对图片中心进行裁剪      
- transforms.ColorJitter      对图像颜色的对比度、饱和度和零度进行变换      
- transforms.FiveCrop     对图像四个角和中心进行裁剪得到五分图像     
- transforms.Grayscale      对图像进行灰度变换    
- transforms.Pad        使用固定值进行像素填充     
- transforms.RandomAffine      随机仿射变换    
- transforms.RandomCrop      随机区域裁剪     
- transforms.RandomHorizontalFlip      随机水平翻转     
- transforms.RandomRotation     随机旋转     
- transforms.RandomVerticalFlip     随机垂直翻转 

#### 常用的数据扩增库   

- [**torchvision** ](https://github.com/pytorch/vision)   

  pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以无缝与torch进行集成；但数据扩增方法种类较少，且速度中等

- [imgaug](https://github.com/aleju/imgaug   )

  imgaug是常用的第三方数据扩增库，提供了多样的数据扩增方法，且组合起来非常方便，速度较快

- [albumentations](https://albumentations.readthedocs.io     )

  是常用的第三方数据扩增库，提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快

## 2.3Torch读取数据

在Pytorch中数据是通过Dataset进行封装，并通过DataLoder进行并行读取。所以我们只需要重载一下数据读取的逻辑就可以完成数据的读取


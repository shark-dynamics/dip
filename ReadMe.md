#### 数字图像处理的一些总结
#### 1. 空间域增强
- 相加平均
- 灰度映射
- 直方图均衡化
- 线性滤波
- 非线性滤波

#### 2. 傅里叶变换频域增强
- 低通（平滑）滤波
- 高通（锐化）滤波

#### 3. 图像复原
- 退化函数
- 逆滤波
- 维纳滤波

#### 4. 图片缩放
- 最近邻
- 双线性
- 双三次

#### 5. 边缘检测
- 普通滤波
- Marr-Hildreth
- Canny

#### 6.形态学
- 腐蚀 膨胀
- 开 闭

### 声明：此项目里的实现很多是自己写的，如增强部分，全是按像素卷积的实现，并没有效率，旨在实现背后的思想。

#### 1.前置
> 按一定统计规律生成一些噪声，用于对图像进行加噪干扰

![](intros/noise.png)

#### 2.相加平均
> 多个噪声图片相加平均

![](intros/add_mean.png)

#### 3.取反
![](intros/gray_inverse.png)

#### 4.分段增强
> 用此分段函数，对图片进行分段增强

![](intros/intro_segment_enhance.png)

> 增强后的结果

![](intros/seg_enhance.png)

#### 5.对数变换
![](intros/log_enhance.png)

#### 6.幂律变换
![](intros/r_enhance.png)

#### 7.灰度切割
![](intros/gray_cut.png)

#### 8.阈值
![](intros/threshold.png)

#### 9.位图切割
![](intros/bitmap_cut.png)

#### 10.直方图均衡化
> 修改前后的照片

![](intros/hist_adj.png)

> 修改前后的直方图

![](intros/hist_intro.png)
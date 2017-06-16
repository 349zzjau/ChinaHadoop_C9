# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

# 代码原作者
__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

# batch size
batch_size = 32   

# 获取MNIST输入数据
# 包含了下载、解压和转换
# 带有输入队列执行器QueueRunner
data = tf.sg_data.Mnist(batch_size=batch_size)

# 提取输入图片数据，原始大小28x28
x = data.train.image

# 压缩为14x14
x_small = tf.image.resize_bicubic(x, (14, 14))
# 通过最邻近算法，再升回28x28
x_nearest = tf.image.resize_images(x_small, (28, 28), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# 生成器generator标签（全为1）
y = tf.ones(batch_size, dtype=tf.sg_floatx)

# 判别器discriminator标签
# 一半标签是1：对应真实图片
# 一半是0：对应生成器图片
y_disc = tf.concat([y, y * 0],0)


# 构建生成器网络generator network
# 基于ESPCN scheme
# http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
with tf.sg_context(name='generator', act='relu', bn=True):
    gen = (x_small
           .sg_conv(dim=32)
           .sg_conv()
           .sg_conv(dim=4, act='sigmoid', bn=False)
           .sg_periodic_shuffle(factor=2))

# 添加图片总结image summary
tf.sg_summary_image(gen)

# 准备训练判别器的正、负样本
# 正样本real：真实数据
# 负样本fake：生成器生成 
x_real_pair = tf.concat([x_nearest, x],3)
x_fake_pair = tf.concat([x_nearest, gen],3)


# 组合判别器输入：
# 真实图片real：真实数据
# 伪造图片fake：生成器生成 
xx = tf.concat([x_real_pair, x_fake_pair],0)


# 构建判别器discriminator
with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu'):
    disc = (xx.sg_conv(dim=64)
              .sg_conv(dim=128)
              .sg_flatten()
              .sg_dense(dim=1024)
              .sg_dense(dim=1, act='linear')
              .sg_squeeze())

# loss函数
loss_disc = tf.reduce_mean(disc.sg_bce(target=y_disc))  # discriminator loss
loss_gen = tf.reduce_mean(disc.sg_reuse(input=x_fake_pair).sg_bce(target=y))  # generator loss
# 优化操作
train_disc = tf.sg_optim(loss_disc, lr=0.0001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops


# 定义训练函数
# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    # 训练判别器discriminator
    l_disc = sess.run([loss_disc, train_disc])[0]
    # 训练生成器generator
    l_gen = sess.run([loss_gen, train_gen])[0]
    return np.mean(l_disc) + np.mean(l_gen)

# 执行训练
alt_train(log_interval=10, max_ep=20, ep_size=data.train.num_batch, early_stop=False)

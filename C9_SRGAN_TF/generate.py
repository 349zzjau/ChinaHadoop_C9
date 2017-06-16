# -*- coding: utf-8 -*-
import sugartensor as tf
import matplotlib.pyplot as plt

# 代码原作者
__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

# batch size
batch_size = 30   

# 获取MNIST输入数据
# 包含了下载、解压和转换
# 带有输入队列执行器QueueRunner
data = tf.sg_data.Mnist(batch_size=batch_size)

# 提取输入图片数据，原始大小28x28
x = data.train.image

# 准备测试数据
x_small = tf.image.resize_bicubic(x, (14, 14))
x_bicubic = tf.image.resize_bicubic(x_small, (28, 28)).sg_squeeze()
x_nearest = tf.image.resize_images(x_small, (28, 28), tf.image.ResizeMethod.NEAREST_NEIGHBOR).sg_squeeze()


# 构建生成器网络generator network
# 基于ESPCN scheme
# http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
with tf.sg_context(name='generator', act='relu', bn=True):
    gen = (x_small
           .sg_conv(dim=32)
           .sg_conv()
           .sg_conv(dim=4, act='sigmoid', bn=False)
           .sg_periodic_shuffle(factor=2)
           .sg_squeeze())


# 执行生成器
fig_name = 'result/sample.png'
with tf.Session() as sess:
    with tf.sg_queue_context(sess):

        tf.sg_init(sess)

        # 读取ckpt文件，调用模型参数
        saver = tf.train.Saver()
        lastest_model_ckpt = tf.train.latest_checkpoint('asset/train')
        saver.restore(sess, lastest_model_ckpt)

        # 执行生成器
        gt, low, bicubic, sr = sess.run([x.sg_squeeze(), x_nearest, x_bicubic, gen])

        # 画结果
        _, ax = plt.subplots(10, 12, sharex=True, sharey=True)
        for i in range(10):
            for j in range(3):
                ax[i][j*4].imshow(low[i*3+j], 'gray')
                ax[i][j*4].set_axis_off()
                ax[i][j*4+1].imshow(bicubic[i*3+j], 'gray')
                ax[i][j*4+1].set_axis_off()
                ax[i][j*4+2].imshow(sr[i*3+j], 'gray')
                ax[i][j*4+2].set_axis_off()
                ax[i][j*4+3].imshow(gt[i*3+j], 'gray')
                ax[i][j*4+3].set_axis_off()
                
        # 存为图片
        plt.savefig(fig_name, dpi=600)
        tf.sg_info('Sample image saved to "%s"' % fig_name)
        plt.close()

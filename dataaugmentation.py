# 数据增强
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import matplotlib.image as mp
import cv2

# 读图
img= cv2.imread('hyun_positive_data\\result1')

# 转为 numpy 数组
data = img_to_array(img)
# 扩展维度
samples = expand_dims(data, 0)
print(samples.shape)
# 创建生成器
datagen = ImageDataGenerator(
        rotation_range=180, #整数，数据提升时图片随机转动的角度
        #width_shift_range=0.08, #浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        #height_shift_range=0.08,#浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        shear_range=0.1,#剪切强度（逆时针方向)
        zoom_range=0.1,#随机缩放的幅度
        fill_mode='nearest') #进行变换时，超出边界的点将根据本参数给定的方法进行处理
# 准备迭代器
it = datagen.flow(samples, batch_size=9,save_format='png')
# 生成图片并画图
for i in range(200):

    batch=it.next()

    image = batch[0].astype('uint8')

    pyplot.imshow(image)

    mp.imsave("hyun_positive_data"%i,image)#保存图片路径
# 展示图片
pyplot.show()





from keras.models import Model
from keras.layers import Input,Conv2D,GlobalAveragePooling2D,Reshape,Dropout
from keras.utils.vis_utils import plot_model
from MobilenetBase import MobilenetBase

class MobileNetV3_Small(MobilenetBase):


    def __init__ (self,shape,n_class,alpha=1.0,include_top=True):
        self.n_class=n_class
        self.shape=shape
        super(MobileNetV3_Small,self).__init__(shape,n_class,alpha)
        self.include_top=include_top


    def build(self,plot=False):
        inputs=Input(shape=self.shape)

        x = self._conv_block(inputs,16,(3,3),strides=(2,2),nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        #x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')#
        #x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')#
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        #x = self._bottleneck(x, 40, (5, 5), e=240, s=2, squeeze=True, nl='HS')#
        #x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')#
        #x = self._bottleneck(x, 48, (5, 5), e=96, s=1, squeeze=True, nl='HS')#
        x = self._bottleneck(x, 96, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        #x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')#
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')

        x=self._conv_block(x,576,(1,1),strides=(1,1),nl='HS')
        x=GlobalAveragePooling2D()(x)  #squeeze
        x=Reshape((1,1,576))(x)
        x =Dropout(0.2,name='dropout')(x)
        #x=Conv2D(1280,(1,1),padding='same')(x)
        x = Conv2D(1280, (1, 1), padding='same')(x)
        x=self._return_activation(x,'HS')

        x=Conv2D(self.n_class,(1,1),padding='same',activation='softmax')(x)

        output=Reshape((self.n_class,))(x)

        model=Model(inputs,output)

        if plot:
            plot_model(model,to_file='C:\\Users\\miryu\\Desktop\\mobilenetv3_small.png',show_shapes='True')
        return model

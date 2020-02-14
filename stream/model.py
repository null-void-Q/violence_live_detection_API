import numpy as np
from keras.layers import Dropout,Dense,GlobalAveragePooling3D
from keras.layers import Reshape
from keras.layers import Lambda
from .i3d_inception import Inception_Inflated3d,conv3d_bn
from keras.models import Model
from keras.layers import Activation
from keras import backend as K
import tensorflow as tf
def loadModel(numberOfClasses,inputFrames, frameDims,withWeights = None , withTop = False):
    graph = tf.Graph()
    with graph.as_default():
        session = tf.Session(graph=graph)
        with session.as_default():
            weights = None
            if withWeights : weights = withWeights
            model = Inception_Inflated3d(
                        include_top=withTop,
                        weights=weights,
                        input_shape=(inputFrames, *frameDims),
                        dropout_prob=0.5,
                        endpoint_logit=False,
                        classes=numberOfClasses,
                        )

            if not withTop:    
                x = model.output
                x = Dropout(0.5)(x)

                x = conv3d_bn(x,numberOfClasses, 1, 1, 1, padding='same', 
                                use_bias=True, use_activation_fn=False, use_bn=False)
                
                num_frames_remaining = int(x.shape[1])
                x = Reshape((num_frames_remaining, numberOfClasses))(x)

                        # logits (raw scores for each class)
                x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                                output_shape=lambda s: (s[0], s[2]))(x)

                predictions = Activation('softmax')(x)
                model = Model(model.input, predictions)

            model._make_predict_function()
            data = np.random.rand(1,64,224,224,3)
            model.predict(data)
            return model,session,graph 
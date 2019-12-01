from keras.layers import Dropout,Dense,GlobalAveragePooling3D
from keras.layers import Reshape
from keras.layers import Lambda
from i3d_inception import Inception_Inflated3d,conv3d_bn
from keras.models import Model
from keras.layers import Activation

def loadModel(numberOfClasses,inputFrames, frameHeight,frameWidth,numRGBChannels,withWeights = False):
    weights = None
    if withWeights : weights = 'rgb_inception_i3d'
    rgb_model = Inception_Inflated3d(
                include_top=False,
                weights=weights,
                input_shape=(inputFrames, frameHeight, frameWidth, numRGBChannels),
                dropout_prob=0.5,
                endpoint_logit=False,
                classes=numberOfClasses)

    x = rgb_model.output
    x = Dropout(0.5)(x)

    x = conv3d_bn(x,numberOfClasses, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
    
    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, numberOfClasses))(x)

            # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                    output_shape=lambda s: (s[0], s[2]))(x)

    predictions = Activation('softmax', name='prediction')(x)
    model = Model(rgb_model.input, predictions)
    
    return model 
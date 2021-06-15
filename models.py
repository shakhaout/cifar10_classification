from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
########################################################################################################################

## Encoder model based on modified VGG16 Architecture
def encoder(input_image): #,**kwargs
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
    x = BatchNormalization(name='block1_BN1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_BN2')(x)
    x = MaxPooling2D((2, 2),  name='block1_pool')(x)
    x= Dropout(0.3, name='block1_dropout')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization(name='block2_BN1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization(name='block2_BN2')(x)
    x = MaxPooling2D((2, 2),  name='block2_pool')(x) 
    x= Dropout(0.3, name='block2_dropout')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization(name='block3_BN1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_BN2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    encoded = BatchNormalization(name='block3_BN3')(x)

    
    return encoded
#######################################################################################################################

## Decoder model based on modified VGG16 Architecture
def decoder(encoded):
    
    # Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(encoded)
    x = BatchNormalization(name='block4_BN1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_BN2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = BatchNormalization(name='block4_BN3')(x)


    # Block 5
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization(name='block5_BN1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization(name='block5_BN2')(x)
    x = UpSampling2D((2, 2), name='block5_pool')(x)

    # Block 6
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = BatchNormalization(name='block6_BN1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = BatchNormalization(name='block6_BN2')(x)
    x = UpSampling2D((2, 2), name='block6_pool')(x)
    decoded = Conv2D(3,(3,3),activation='sigmoid',padding='same', name='block6_conv3')(x)
    
    return decoded
#######################################################################################################################

## CNN Classification model
def classifier(inputs):
    
    # Block 7
    x = Conv2D(64, (3, 3), padding='same', name='block7_conv1')(inputs)
    x = BatchNormalization(name='block7_BN1')(x)
    x = Activation('relu', name='block7_activation1')(x)
    x = MaxPooling2D((2, 2),  name='block7_pool1')(x) 
    
    
    # Block 8
    x = Conv2D(64, (3, 3), padding='same', name='block8_conv1')(x)
    x = BatchNormalization(name='block8_BN1')(x)
    x = Activation('relu', name='block8_activation1')(x)
    x = MaxPooling2D((2,2), name='block8_pool1')(x)
    x = Flatten(name='block8_flatten')(x)
    x = Dense(512, activation='relu', name='block8_dense1')(x)
    x = Dropout(0.4, name='block8_dropout2')(x)
    x = Dense(64, activation='relu', name='block8_dense2')(x)
    out = Dense(10, activation='softmax', name='block8_dense3')(x)
    
    return out

###################################################################################################################

## Modified AutoEncoder CNN model

def EnCNN_cls(inputs):
    x = Flatten(name='block8_flatten')(inputs)
    x = Dense(512, activation='relu', name='block8_dense1')(x)
    x = Dropout(0.4, name='block8_dropout2')(x)
    x = Dense(64, activation='relu', name='block8_dense2')(x)
    out = Dense(10, activation='softmax', name='block8_dense3')(x)

    return out
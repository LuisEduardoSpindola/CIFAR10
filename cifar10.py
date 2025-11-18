import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, AveragePooling2D,
    Flatten, Dense, Dropout, Activation, Add
)

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

def build_lenet():
    """
    Constrói um modelo LeNet-5 adaptado para Cifar-10 (32x32x3).
    """
    model = Sequential(name="LeNet")

    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=INPUT_SHAPE))

    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))

    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    return model

def build_minivgg():
    """
    Constrói a arquitetura MiniVGG exatamente como especificada na imagem.
    """
    model = Sequential(name="MiniVGG")

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    return model

def build_mini_resnet():
    """
    Constrói uma Mini-ResNet. Esta arquitetura usa a API Funcional
    do Keras para criar as conexões residuais (skip connections).
    """

    inputs = Input(shape=INPUT_SHAPE)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = Activation('relu')(x)

    shortcut = x

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    shortcut = Conv2D(64, (1, 1), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Mini-ResNet")

    return model

print("Instanciando LeNet...")
model_lenet = build_lenet()
model_lenet.summary()

print("\n" + "="*50 + "\n")

print("Instanciando MiniVGG...")
model_minivgg = build_minivgg()
model_minivgg.summary()

print("\n" + "="*50 + "\n")

print("Instanciando Mini-ResNet...")
model_resnet = build_mini_resnet()
model_resnet.summary()
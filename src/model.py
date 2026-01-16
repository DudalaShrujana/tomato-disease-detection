from tensorflow.keras import layers, models

def build_model():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu',
             input_shape=(224,224,3)))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # 11 classes → softmax
    model.add(layers.Dense(11, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


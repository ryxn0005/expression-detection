from keras import layers, models

def build_convnet(
    n_classes: int = 7,
    drop_out: float = 0.25,
    input_shape=(48, 48, 1),
):
    return models.Sequential(
        [
            layers.Input(shape=input_shape),

            # Block 1
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Block 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Block 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Block 4
            layers.Conv2D(256, (3, 3), padding="same"),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Dense head
            layers.Flatten(),
            layers.Dense(256),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dropout(drop_out*2),
            layers.Dense(512),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dropout(drop_out*2),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )

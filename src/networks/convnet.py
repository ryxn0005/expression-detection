from keras import layers, models

def build_convnet(
    n_classes: int = 9,
    drop_out: float = 0.25,
    input_shape=(64, 64, 1),
):
    return models.Sequential(
        [
            layers.Input(shape=input_shape),

            # Block 1
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Block 2
            layers.Conv2D(128, (5, 5), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Block 3
            layers.Conv2D(512, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Block 4
            layers.Conv2D(512, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Dense head
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(drop_out),

            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(drop_out),

            layers.Dense(n_classes, activation="softmax"),
        ]
    )

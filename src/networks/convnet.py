from keras import layers, models, regularizers

def build_convnet(
    n_classes: int = 7,
    drop_out: float = 0.25,
    input_shape: tuple = (48, 48, 1),
    l2_weight: float = 0.01,
):
    """
    Build a ConvNet for facial expression classification (Sequential API).

    @Usage:
        model = build_convnet(n_classes=7, input_shape=(48, 48, 1), l2_weight=0.01)
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

    @Parameters:
        n_classes : int
            Number of output classes. Defaults to 7.
        input_shape : tuple[int, int, int]
            Input tensor shape as (H, W, C). Use (48, 48, 1) for grayscale or (48, 48, 3) for RGB.
        l2_weight : float
            L2 regularization strength applied to the 128- and 256-filter conv layers.

    @Returns:
        keras.Model
            A compiled-ready Keras Sequential model with:
            [Conv(32)->ReLU, Conv(64)->ReLU, BN, MaxPool, Dropout(0.25),
             Conv(128,l2)->ReLU, Conv(256,l2)->ReLU, BN, MaxPool, Dropout(0.25),
             Flatten, Dense(1024)->ReLU, Dropout(0.5), Dense(n_classes)->Softmax]
    """
    l2_reg = regularizers.l2(l2_weight)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            # Block 1
            layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Block 2
            layers.Conv2D(128, (3, 3), padding="same",
                          kernel_regularizer=l2_reg, kernel_initializer="he_normal"),
            layers.ReLU(),
            layers.Conv2D(256, (3, 3), padding="same",
                          kernel_regularizer=l2_reg, kernel_initializer="he_normal"),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(drop_out),

            # Dense head
            layers.Flatten(),
            layers.Dense(1024, kernel_initializer="he_normal"),
            layers.ReLU(),
            layers.Dropout(drop_out*2),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    return model


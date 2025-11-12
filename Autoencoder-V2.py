# ======================================================
# ADVANCED AUTOENCODER WITH MULTI-SCALE, RESIDUAL, LSTM & CBAM
# ======================================================

def build_autoencoder(input_shape=(WIN_SAMPLES, 1), latent_dim=20):
    """
    Deep hybrid autoencoder for EEG feature extraction:
    - Multi-scale convolutions
    - Residual + CBAM + SE blocks
    - BiLSTM bottleneck
    - Symmetric Conv1DTranspose decoder
    """
    from tensorflow.keras import layers, models, backend as K

    def conv_block(x, filters, kernel_size, strides=1):
        """Conv1D block with BatchNorm and GELU"""
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)
        return x

    def se_block(inputs, ratio=8):
        """Squeeze-and-Excitation block"""
        filters = inputs.shape[-1]
        se = layers.GlobalAveragePooling1D()(inputs)
        se = layers.Dense(filters // ratio, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Multiply()([inputs, layers.Reshape((1, filters))(se)])
        return se

    def cbam_block(inputs, ratio=8):
        """Channel + spatial attention (CBAM 1D)"""
        channel = inputs.shape[-1]
        shared_dense_one = layers.Dense(channel // ratio, activation='relu', use_bias=False)
        shared_dense_two = layers.Dense(channel, activation='sigmoid', use_bias=False)
        avg_pool = layers.GlobalAveragePooling1D()(inputs)
        max_pool = layers.GlobalMaxPooling1D()(inputs)
        avg_dense = shared_dense_two(shared_dense_one(avg_pool))
        max_dense = shared_dense_two(shared_dense_one(max_pool))
        channel_attention = layers.Add()([avg_dense, max_dense])
        channel_attention = layers.Activation('sigmoid')(channel_attention)
        channel_refined = layers.Multiply()([inputs, layers.Reshape((1, channel))(channel_attention)])

        # Spatial attention
        avg_pool = K.mean(channel_refined, axis=-1, keepdims=True)
        max_pool = K.max(channel_refined, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = layers.Conv1D(1, 7, padding='same', activation='sigmoid')(concat)
        refined = layers.Multiply()([channel_refined, spatial_attention])
        return refined

    # ---------------- ENCODER ----------------
    inp = layers.Input(shape=input_shape)

    # Multi-scale initial block
    conv3 = conv_block(inp, 32, 3)
    conv5 = conv_block(inp, 32, 5)
    conv7 = conv_block(inp, 32, 7)
    x = layers.Concatenate()([conv3, conv5, conv7])
    x = layers.Conv1D(64, 1, padding='same', activation='gelu')(x)

    # Residual downsampling blocks
    skip1 = x
    x = conv_block(x, 128, 5, strides=2)
    x = se_block(x)
    skip2 = x
    x = conv_block(x, 256, 3, strides=2)
    x = cbam_block(x)

    # BiLSTM bottleneck
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)
    latent = layers.Dense(latent_dim, activation=None, name='latent')(x)

    # ---------------- DECODER ----------------
    x = layers.Dense((WIN_SAMPLES // 4) * 128, activation='gelu')(latent)
    x = layers.Reshape((WIN_SAMPLES // 4, 128))(x)

    x = layers.Conv1DTranspose(128, 3, strides=2, padding='same', activation='gelu')(x)
    x = layers.Add()([x, skip2])  # residual connection
    x = layers.Conv1DTranspose(64, 5, strides=2, padding='same', activation='gelu')(x)
    x = layers.Add()([x, skip1])  # residual connection

    out = layers.Conv1D(1, 3, padding='same', activation='linear')(x)

    # ---------------- MODEL ----------------
    model = models.Model(inp, out, name='DeepHybrid_CBAM_Autoencoder')
    encoder = models.Model(inp, latent, name='DeepHybrid_Encoder')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

    return model, encoder

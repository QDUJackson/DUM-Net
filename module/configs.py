import ml_collections

def get_2DReg_config():
    config = ml_collections.ConfigDict()

    config.patches = ml_collections.ConfigDict({'size': (8, 8)})

    config.patches.grid = (8, 8)

    config.hidden_size = 128   # 252 128
    config.transformer = ml_collections.ConfigDict()

    config.transformer.mlp_dim = 1024  #3072 512

    config.transformer.num_heads = 8 # 12 8
    config.transformer.num_layers = 8 # 12 从6调成2 从2调到1  2

    config.transformer.attention_dropout_rate = 0.0

    config.transformer.dropout_rate = 0.1
    config.patch_size = 8

    config.conv_first_channel = 256 # 512 128 256

    config.encoder_channels = (16, 32, 32)

    config.down_factor = 2

    config.down_num = 2

    config.decoder_channels = (64,32,16,16,8)  #(96, 48, 32, 32, 16)  (64,32,16,16,8)

    config.skip_channels = (32, 32, 32, 32, 16)
    config.n_dims = 2
    config.n_skip = 5
    return config

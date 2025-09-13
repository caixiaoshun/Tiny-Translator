class Config:
    def __init__(self):
        self.encoder_layer = 6
        self.decoder_layer = 6
        self.embed_dim = 512
        self.num_heads = 8
        self.drop_out = 0.1
        self.max_len = 256
        self.vocab_size = 30_000
        self.wmt_zh_en_path = "data/wmt_zh_en_training_corpus.csv"
        self.tokenizer_file = "checkpoints/tokenizer.json"
        self.batch_size = 64
        self.compile = False
        self.seed = 42
        self.val_ratio = 0.1
        self.num_workers = 4
        self.pin_memory = True
        self.tensorboard_dir = "log/tensorboard"
        self.checkpoint_dir = "log/checkpoint"

        self.base_lr = 3e-4
        self.betas = (0.9, 0.98)
        self.eps = 1e-9
        self.weight_decay = 0.1
        self.warmup_ratio = 0.005
        self.start_factor = 1e-3
        self.end_factor = 1.0
        self.eta_min = 3e-6
        self.max_epochs = 1

        self.data_cache_dir = "data/cache.pickle"
        self.use_cache = True

        self.every_n_train_steps = 10000
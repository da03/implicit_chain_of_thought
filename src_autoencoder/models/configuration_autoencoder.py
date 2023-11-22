from transformers import PretrainedConfig

class AutoEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        mixture_size=1,
        teacher_hidden_size=None,
        teacher_num_layers=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size
        self.teacher_hidden_size = teacher_hidden_size
        self.teacher_num_layers = teacher_num_layers
        super().__init__(**kwargs)

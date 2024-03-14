from transformers import PretrainedConfig

class StudentConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        mixture_size=1,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model
        self.mixture_size = mixture_size
        super().__init__(**kwargs)

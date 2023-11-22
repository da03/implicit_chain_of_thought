from transformers import PretrainedConfig

class EmulatorConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        mixture_size=1,
        softmax_temperature=0.05,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size
        self.softmax_temperature = softmax_temperature
        super().__init__(**kwargs)


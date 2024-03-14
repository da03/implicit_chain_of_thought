from transformers import PretrainedConfig

class EmulatorConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        mixture_size='1',
        softmax_temperature=0.05,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model 
        self.mixture_size = mixture_size
        self.softmax_temperature = softmax_temperature
        super().__init__(**kwargs)


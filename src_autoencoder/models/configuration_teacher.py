from transformers import PretrainedConfig

class TeacherConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        super().__init__(**kwargs)


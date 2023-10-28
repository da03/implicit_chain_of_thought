import torch
from transformers import StoppingCriteria

def get_sep_position(input_ids, sep_id):
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        sep_position = input_ids[batch_id].eq(sep_id).nonzero()[0, -1].item()
        sep_positions[batch_id] = sep_position
    return sep_positions


# Stop generation only after generating two EOSs, such as  z <eos> y <eos>
class TwoEOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        input_ids = input_ids[0]
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        return eos_count - self.eos_count_init >= 2

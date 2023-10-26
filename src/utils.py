
def get_sep_position(input_ids, sep_id):
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        sep_position = input_ids[batch_id].eq(sep_id).nonzero()[0, -1].item()
        sep_positions[batch_id] = sep_position
    return sep_positions

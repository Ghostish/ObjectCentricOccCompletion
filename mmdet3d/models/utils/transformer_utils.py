import torch

def get_future_mask(L, device, window_size=-1):
    # do not attend to the future
    mask = torch.ones(L, L, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    if window_size > 0:
        for i in range(window_size - 1, L):
            mask[i, :i - window_size + 1] = 1
    return mask
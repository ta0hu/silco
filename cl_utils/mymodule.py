import torch


def sim_matrix(_a, _b, eps=1e-8):
    """
    a: BxT1xC
    b: BxT2xC
    added eps for numerical stability
    return BxT1xT2, means similarity score
    https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    """
    bb, _, _ = _a.shape
    sim_mt_list = []
    for _bb in range(bb):
        a = _a[_bb]
        b = _b[_bb]
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        sim_mt_list.append(sim_mt.unsqueeze(0))

    return torch.cat(sim_mt_list, 0)


def sample_weighted_cross_entropy():
    pass

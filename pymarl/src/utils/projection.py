import torch as th

def proj(next_v_range, prob, v_min, v_max, n_atom):
    # next_v_range: [bs, episode_len, ..., n]
    # prob: [bs, episode_len, ..., n]
    v_step = (v_max - v_min) / (n_atom - 1)
    bs = list(prob.shape[:-1]) #[bs, episode_len, ...,]
    if len(next_v_range.shape) == 1:
        for _ in range(len(bs)):
            next_v_range = next_v_range.unsqueeze(0)
        next_v_range = next_v_range.expand(*bs, -1)
    proj_prob = th.zeros((*bs, n_atom), device=prob.device) #[bs, episode_len, ..., n_atom]
    # calc relative position of possible value
    next_v_pos = (next_v_range - v_min) / v_step  #[bs, episode_len, ..., n]
    next_v_pos = th.clamp(next_v_pos, 0, n_atom - 1)
    # get lower/upper bound of relative position
    low_bd = next_v_pos.floor().to(th.int64)
    up_bd = next_v_pos.ceil().to(th.int64)
    low_bd[(up_bd > 0) * (low_bd == up_bd)] -= 1
    up_bd[(up_bd == 0) * (low_bd == up_bd)] += 1

    #self[i][j][index[i][j][k]] += src[i][j][k]
    proj_prob.scatter_add_(dim=-1, index=low_bd, src=prob * (up_bd - next_v_pos))
    proj_prob.scatter_add_(dim=-1, index=up_bd, src=prob * (next_v_pos - low_bd))

    return proj_prob #[bs, episode_len, ..., n_atom]
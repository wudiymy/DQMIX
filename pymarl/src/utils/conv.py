import torch as th

def bconv(x, y):
    #x: [bs, episode_len, m]
    #y: [bs, episode_len, n]
    bs = x.size(0)
    episode_len = x.size(1)
    m = x.size(2)
    n = y.size(2)
    x = x.view(bs*episode_len, -1) #[bs, m]
    y = y.view(bs*episode_len, 1, -1) #[bs, 1, n]
    y = y.flip(dims=[2])
    x = th.nn.ConstantPad1d(n-1, 0)(x) #[bs, m+2n-2]
    x = x.unfold(1, n, 1)        #[bs, m+n-1, n]
    x = (x * y).sum(dim=-1) #[bs, m+n-1]
    x = x.view(bs, episode_len, -1)
    return x
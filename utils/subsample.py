import torch
from einops import repeat
from pytorch3d.ops import sample_farthest_points


def create_sampler(sampler='random', **kwargs):

    if sampler == 'random':
        return random_sample
    elif sampler == 'fps':
        return fps_sample
    elif sampler == 'visible':
        return visible_sample
    elif sampler == 'trunc':
        return trunc_sample
    else:
        raise NotImplementedError(
            f'sampler {sampler} not implemented')


def trunc_sample(xyz, n_samples, **kwargs):

    xyz_idx = torch.arange(0, n_samples, dtype=torch.long, device=xyz.device)
    xyz_idx = repeat(xyz_idx, 'n -> b n', b=xyz.shape[0])
    xyz_sampled = xyz[:, :n_samples, :]
    return xyz_sampled, xyz_idx


def fps_sample(xyz, n_samples, **kwargs):
    """
    :param xyz: [B, N, 3]
    :return: [B, n_samples, 3], [B, n_samples]
    """
    random_start_point = kwargs.get('random_start', False)
    xyz_sampled, xyz_idx = sample_farthest_points(
        points=xyz,
        K=n_samples,
        random_start_point=random_start_point,
    )
    return xyz_sampled, xyz_idx


def random_sample(xyz, n_samples, **kwargs):
    """
    :param xyz: [B, N, 3]
    :return: [B, n_samples, 3], [B, n_samples]
    """
    B, N, _ = xyz.shape
    xyz_idx = torch.randint(0, N, (B, n_samples), device=xyz.device)
    xyz_sampled = torch.gather(xyz, 1, xyz_idx.unsqueeze(-1).expand((-1, -1, 3)))
    return xyz_sampled, xyz_idx


def visible_sample(xyz, visible, n_samples, **kwargs):
    """
    :param xyz: [B, N, 3]
    :param visible: [B, N, 1, n_cameras*2]
    :return: [B, n_samples, 3], [B, n_samples]
    """
    B, N, _ = xyz.shape
    n_cameras = visible.shape[-1]
    visible = visible.sum(dim=-1, keepdim=False).squeeze(-1)
    visible = visible / n_cameras
    xyz_idx = torch.multinomial(visible.float().softmax(dim=1), n_samples, replacement=False)
    xyz_sampled = torch.gather(xyz, 1, xyz_idx.unsqueeze(-1).expand((-1, -1, 3)))
    return xyz_sampled, xyz_idx

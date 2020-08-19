import torch

def get_memory_loss(memory_att):
    """The memory attribute should be with size [batch_size, memory_dim, reduced_time_dim, f_h, f_w]
    loss = \sum_{t=1}^{reduced_time_dim} (-mem) * (mem + 1e-12).log() 
    averaged on each pixel and each batch
    2. average over batch_size * fh * fw
    """
    s = memory_att.shape
    memory_att = (-memory_att) * (memory_att + 1e-12).log()  # [batch_size, memory_dim, time, fh, fw]
    memory_att = memory_att.sum() / (s[0] * s[-2] * s[-1]) 
    return memory_att 
    
    
def get_unormalized_data(x_input, x_recons, mean, std):
    x_input = x_input.mul(std).add(mean)
    x_recons = x_recons.mul(std).add(mean)
    return x_input, x_recons


def get_reconstruction_loss(x_input, x_recons, mean=0.5, std=0.5):
    """Calculates the reconstruction loss between x_input and x_recons
    x_input: [batch_size, ch, time, imh, imw]
    x_recons: [batch_size, ch, time, imh, imw]
    """
    batch_size, ch, time_dimension, imh, imw = x_input.shape
    x_input, x_recons = get_unormalized_data(x_input, x_recons, mean, std)
    recons_loss = (x_input - x_recons) ** 2
    recons_loss = recons_loss.sum().sqrt()/(batch_size * imh * imw)
    return recons_loss
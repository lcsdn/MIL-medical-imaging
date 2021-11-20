import torch
from torch import nn

from .attention import AttentionPooling

def old_mean_aggregator(clf_outputs, num_samples): # replaced by MeanAggregator (must change corresponding models because single-dimensional output is now of shape Nx1 instead of N) 
    batch_size = len(num_samples)
    aggregated_outputs = []
    idx_start = 0
    for i, num_image in enumerate(num_samples):
        aggregated_outputs.append(clf_outputs[idx_start:idx_start+num_image].mean(axis=0))
        idx_start += num_image
    return torch.cat(aggregated_outputs)

class Aggregator(nn.Module):
    """
    Basic structure for aggregator block.
    Aggregate bag of samples thanks to a pooling operation defined by user.
    
    The user should define one block:
        pooling: (function or nn.Module) take as input a bag of samples
            (LxD Tensor, where L is the size of the bag) and pool them
            together in a single representation (1xD Tensor).
    """
    
    def forward(self, samples, bags_num_samples):
        """
        Args:
            samples: (Tensor) batch of concatenated bags of samples.
            bags_num_samples: (Tensor) number of samples in each bag.
        
        Output:
            aggregated_samples: (Tensor) bag-level representations.
        """
        batch_size = len(bags_num_samples)
        aggregated_samples = []
        idx_start = 0
        for i, num_samples in enumerate(bags_num_samples):
            bag_samples = samples[idx_start:idx_start+num_samples]
            attended_samples = self.pooling(bag_samples)
            aggregated_samples.append(attended_samples)
            idx_start += num_samples
        return torch.cat(aggregated_samples)
    
class MeanAggregator(Aggregator):
    """Pool the samples in each bag by their average."""
    @staticmethod
    def pooling(bag_samples):
        return bag_samples.mean(axis=0).unsqueeze(0)

class AttentionAggregator(Aggregator):
    """
    Pool the samples in each bag by attending to specific samples thanks to an
    attention mechanism.
    """
    def __init__(self, embed_dim, latent_dim, gated=False):
        super().__init__()
        self.pooling = AttentionPooling(embed_dim, latent_dim, gated=gated)
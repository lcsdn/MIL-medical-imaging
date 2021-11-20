from torch import nn

class AttentionPooling(nn.Module):
    """Pool embeddings using a learned attention mechanism."""
    
    def __init__(self, embed_dim, latent_dim, gated=False):
        """
        Args:
            embed_dim: (int) dimension of input embeddings.
            latent_dim: (int) latent dimension of queries and keys.
            gated: (bool) if True, multiply element-wise the queries by a learned
                sigmoid non-linearity.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.gated = gated
        self.query_layer = nn.Linear(self.embed_dim, self.latent_dim)
        self.key_layer = nn.Linear(latent_dim, 1)
        if self.gated:
            self.gate_layer = nn.Linear(self.embed_dim, self.latent_dim)

    def forward(self, embeddings):
        query = self.query_layer(embeddings)
        query = query.tanh()
        if self.gated:
            gate = self.gate_layer(embeddings)
            gate = gate.sigmoid()
            query *= gate
        scores = self.key_layer(query)
        attention = scores.softmax(1)
        attended_embeddings = (attention.T @ embeddings)
        return attended_embeddings

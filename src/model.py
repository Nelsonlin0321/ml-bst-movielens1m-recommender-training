from torch import nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self, input_size, output_size, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(input_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, output_size),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm2 = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Multi-head Attention
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Feed-Forward Network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout_rate=0.2, num_layers=3):
        super(TransformerLayer, self).__init__()

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_model, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x


class MLP(nn.Module):
    def __init__(self, dropout=0.2, hidden_units=[512, 256, 128]):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=dropout))
        self.fc = nn.Linear(hidden_units[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        logits = self.fc(x)
        output = self.sigmoid(logits)
        return output


class BSTRecommenderModel(nn.Module):
    def __init__(self, config):
        super(BSTRecommenderModel, self).__init__()

        self.config = config
        self.embed_configs = config.embed_configs
        self.drouput = config.dropout
        self.transformer_num_layer = config.transformer_num_layer
        self.device = config.device

        embed_configs = self.config.embed_configs

        """Create Embedding Layer"""
        embedding_layers = []
        for name, embed_config in embed_configs.items():
            embed_dim = embed_config['embed_dim']
            num_embed = embed_config['num_embed']
            embeding_layer = nn.Embedding(
                num_embeddings=num_embed, embedding_dim=embed_dim)
            nn.init.xavier_uniform_(embeding_layer.weight)
            embedding_layers.append([name, embeding_layer])

        self.embedding_layers = nn.ModuleDict(embedding_layers)

        transformer_dim = self.embed_configs['position']['embed_dim'] + \
            self.embed_configs['movie']['embed_dim'] + \
            self.embed_configs['genre']['embed_dim']

        self.transformer_layer = TransformerLayer(d_model=transformer_dim,
                                                  num_heads=8,
                                                  dropout_rate=self.drouput,
                                                  num_layers=self.transformer_num_layer)

        user_features_dim = 1+self.embed_configs['movie']['embed_dim'] + \
            self.embed_configs['age_group']['embed_dim']*2

        sequence_length = self.embed_configs['position']['num_embed']

        mlp_dim = transformer_dim*sequence_length + user_features_dim
        self.mlp = MLP(dropout=self.drouput, hidden_units=[mlp_dim, 256, 64])

    def forward(self, inputs):

        target_movie_embedding = self.embedding_layers['movie'](
            inputs['target_movie'])
        batch_size = target_movie_embedding.shape[0]

        movie_sequence_embedding = self.embedding_layers['movie'](
            inputs['movie_sequence'])

        # genres
        genres_sequence_emebdding = self.embedding_layers['genre'](
            inputs['genres_ids_sequence'])

        # genres max pooling
        genres_sequence_emebdding = torch.mean(
            genres_sequence_emebdding, dim=-2)
        genres_sequence_cross_target_movie_emebdding = torch.mul(
            genres_sequence_emebdding, target_movie_embedding.view(
                batch_size, 1, -1)
        )

        # position embedding
        positions = torch.arange(
            self.config.embed_configs['position']['num_embed']).to(self.device)
        position_embedding = self.embedding_layers['position'](positions)
        batch_position_embedding = torch.stack(
            [position_embedding.clone() for _ in range(batch_size)])

        movie_pos_genres_seq_embedding = torch.concat(
            [movie_sequence_embedding,
             genres_sequence_cross_target_movie_emebdding,
             batch_position_embedding], dim=-1)

        seq_transformer_output = self.transformer_layer(
            movie_pos_genres_seq_embedding)

        seq_transformer_flatten_output = seq_transformer_output.view(
            batch_size, -1)

        #  """concat other features"""
        sex_feature = inputs['sex'].unsqueeze(-1)
        age_group_embedding = self.embedding_layers['age_group'](
            inputs['age_group_index'])

        sex_cross_feature = torch.mul(sex_feature, target_movie_embedding)

        age_group_embedding_cross = torch.mul(
            age_group_embedding, target_movie_embedding)

        user_input_features = torch.concat(
            [sex_feature, sex_cross_feature, age_group_embedding, age_group_embedding_cross], dim=-1)

        mlp_input_features = torch.concat(
            [user_input_features, seq_transformer_flatten_output], dim=1)

        outputs = self.mlp(mlp_input_features)

        return outputs

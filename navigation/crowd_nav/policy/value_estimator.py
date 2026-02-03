import torch.nn as nn
from crowd_nav.policy.helpers import mlp


class ValueEstimator(nn.Module):
    def __init__(self, config, graph_model):
        super().__init__()
        self.graph_model = graph_model
        self.value_network = mlp(config.gcn.X_dim, config.model_predictive_rl.value_network_dims)

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        args:
            state: (robot_state, human_states)
                    robot_state: [batch_size, 1, robot_state_dim], human_states: [batch_size, num_human, human_state_dim]
        returns:
            value: [batch_size, 1]
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        # only use the feature of robot node as state representation
        state_embedding = self.graph_model(state)[:, 0, :]
        value = self.value_network(state_embedding)
        return value

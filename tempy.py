import numpy as np
from ray.rllib.models.catalog import ModelCatalog
import gym

env = gym.make("CartPole-v1")

model = ModelCatalog.get_model_v2(
        obs_space=env.observation_space,
        action_space=env.action_space,
        num_outputs=np.array(1),
        framework="tf",
        model_config={
            "vf_share_layers": False,

            # "use_lstm": True,
            # "max_seq_len": 32,
            # "lstm_cell_size": 128,
            # "lstm_use_prev_action": True,

            "use_attention": True,
            "max_seq_len": 30,
            "attention_num_transformer_units": 1,
            "attention_dim": 32,
            "attention_memory_inference": 100,
            "attention_memory_training": 50,
            "attention_num_heads": 8,
            "attention_head_dim": 32,
            "attention_position_wise_mlp_dim": 256,
            "attention_use_n_prev_actions": 0,
            "attention_use_n_prev_rewards": 0,
            "attention_init_gru_gate_bias": 2.0
    },
    )

print(model.summary())
print('-----------------')
# print(model.gtrxl.summary())
print('wowowowowowowowowowo')
print(model.gtrxl.trxl_model.summary())
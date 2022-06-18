from gym import spaces
import ray
from ray.rllib.agents import with_common_config
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import PolicyServerInput

from ray.rllib.examples.env.random_env import RandomEnv

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-ip', type=str, help='IP of this device')

parser.add_argument('-checkpoint', type=str, help='location of checkpoint to restore from')

args = parser.parse_args()

DEFAULT_CONFIG = with_common_config({
    "gamma": 0.996,
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE (lambda) parameter.
    "lambda": 0.98,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Target value for KL divergence.
    "kl_target": 0.02,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 16,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 4096,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 256,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 1,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": False,
    # Stepsize of SGD.
    "lr": 3e-5,
    # Learning rate schedule.
    "lr_schedule": None,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    "vf_loss_coeff": 0.01,
    "model": {
        # Share layers for value function. If you set this to True, it's
        # important to tune vf_loss_coeff.
        "vf_share_layers": True,

        # "use_lstm": True,
        # "max_seq_len": 32,
        # "lstm_cell_size": 128,
        # "lstm_use_prev_action": True,

        'use_attention': True,
        "max_seq_len": 125,
        "attention_num_transformer_units": 1,
        "attention_dim": 512,
        "attention_memory_inference": 125,
        "attention_memory_training": 125,
        "attention_num_heads": 8,
        "attention_head_dim": 64,
        "attention_position_wise_mlp_dim": 256,
        "attention_use_n_prev_actions": 0,
        "attention_use_n_prev_rewards": 0,
        "attention_init_gru_gate_bias": 2.0,
        # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
        # These are used if no custom model is specified and the input space is 2D.
        # Filter config: List of [out_channels, kernel, stride] for each filter.
        # Example:
        # Use None for making RLlib try to find a default filter setup given the
        # observation space.
        "conv_filters": [
            # [4, [3, 4], [1, 1]],
            # [16, [6, 8], [3, 3]],
            # [32, [6, 8], [3, 4]],
            # [64, [6, 6], 3],
            # [256, [9, 9], 1],

            #480 x 640
            # [4, [7, 7], [3, 3]],
            # [16, [5, 5], [3, 3]],
            # [32, [5, 5], [2, 2]],
            # [64, [5, 5], [2, 2]],
            # [256, [5, 5], [3, 5]],

            # 240 X 320
            # [64, [12, 16], [7, 9]],
            # [128, [6, 6], 4],
            # [256, [9, 9], 1]
        ],
        # A,
        # Activation function descriptor.
        # Supported values are: "tanh", "relu", "swish" (or "silu"),
        # "linear" (or None).
        # "conv_activation": "relu",

        # Some default models support a final FC stack of n Dense layers with given
        # activation:
        # - Complex observation spaces: Image components are fed through
        #   VisionNets, flat Boxes are left as-is, Discrete are one-hot'd, then
        #   everything is concated and pushed through this final FC stack.
        # - VisionNets (CNNs), e.g. after the CNN stack, there may be
        #   additional Dense layers.
        # - FullyConnectedNetworks will have this additional FCStack as well
        # (that's why it's empty by default).
        # "post_fcnet_hiddens": [256, 256],
        # "post_fcnet_activation": "relu",
    },
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.00005,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.2,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 5.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "complete_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # # the default optimizer.
    "simple_optimizer": True,
    # "reuse_actors": True,
    "num_gpus": 0,
    # Use the connector server to generate experiences.
    "input": (
        lambda ioctx: PolicyServerInput(ioctx, args.ip, 55556)
    ),
    # Use a single worker process to run the server.
    "num_workers": 0,
    # Disable OPE, since the rollouts are coming from online clients.
    "input_evaluation": [],
    # "callbacks": MyCallbacks,
    "env": RandomEnv,
    "env_config": {
        "sleep": True
    },
    "framework": "tf",
    # "eager_tracing": True,
    "explore": True,
    # "exploration_config": {
    #     "type": "Curiosity",  # <- Use the Curiosity module for exploring.
    #     "eta": 0.6,  # Weight for intrinsic rewards before being added to extrinsic ones.
    #     "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
    #     "feature_dim": 1152, # Dimensionality of the generated feature vectors.
    #     # Setup of the feature net (used to encode observations into feature (latent) vectors).
    #     "inverse_net_hiddens": [64, 128], # Hidden layers of the "inverse" model.
    #     "inverse_net_activation": "relu",  # Activation of the "inverse" model.
    #     "forward_net_hiddens": [64, 128],  # Hidden layers of the "forward" model.
    #     "forward_net_activation": "relu",  # Activation of the "forward" model.
    #     "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
    #     # Specify, which exploration sub-type to use (usually, the algo's "default"
    #     # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
    #     "sub_exploration": {
    #         "type": "StochasticSampling",
    #     }
    # },
    "create_env_on_driver": False,
    "log_sys_usage": False,
    # "normalize_actions": False,
    "compress_observations": True,
    # Whether to fake GPUs (using CPUs).
    # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
    "_fake_gpus": False
})

x = 320
y = 240

# x = 640
# y = 480

DEFAULT_CONFIG["env_config"]["observation_space"] = spaces.Box(low=0, high=1, shape=(y, x, 1), dtype=np.float32)

DEFAULT_CONFIG["env_config"]["action_space"] = spaces.MultiDiscrete(
    [
        2,  # W
        2,  # A
        2,  # S
        2,  # D
        2,  # Space
        2,  # H
        2,  # J
        2,  # K
        2  # L
    ]
)

# ray.init(log_to_driver=False)
ray.init(num_cpus=5, num_gpus=0, log_to_driver=False)

trainer = PPOTrainer

from ray import tune

name = "" + args.checkpoint
print(f"Starting: {name}")
tune.run(trainer,
         resume = 'AUTO',
         config=DEFAULT_CONFIG, name=name, keep_checkpoints_num=None, checkpoint_score_attr="episode_reward_mean",
         max_failures=99,
         # restore="C:\\Users\\denys\\ray_results\\mediumbrawl-attention-256Att-128MLP-L2\\PPOTrainer_RandomEnv_1e882_00000_0_2022-06-02_15-13-44\\checkpoint_000028\\checkpoint-28",
         checkpoint_freq=1, checkpoint_at_end=True)

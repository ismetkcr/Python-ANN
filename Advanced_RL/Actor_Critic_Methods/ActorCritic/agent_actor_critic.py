import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow.keras as keras
from network_actor_critic import ActorCriticNetwork


class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='models/'):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        self.checkpoint_file = os.path.join(chkpt_dir, '_actor_critic')

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)
    
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        
        # Ensure action is within the valid range of action space
        self.action = action.numpy()[0]
        
        # Clip action to be within the valid action space
        if self.action >= self.n_actions:
            self.action = self.n_actions - 1
        
        return self.action


    def save_models(self):
        self.actor_critic.save(self.checkpoint_file)
        print('... saving models ...')

    def load_models(self):
        self.actor_critic = keras.models.load_model(self.checkpoint_file)
        print('... loading models ...')

    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + \
                self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
        params = self.actor_critic.trainable_variables
        grads = tape.gradient(total_loss, params)
        self.actor_critic.optimizer.apply_gradients(zip(grads, params))
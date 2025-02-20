import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from network_ddqn import DeepQNetwork
from replay_memory_ddqn import ReplayBuffer


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/ddqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.fname = self.chkpt_dir + self.env_name + '_' + self.algo + '_'

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(input_dims, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DeepQNetwork(input_dims, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        self.q_eval.save(self.fname+'q_eval')
        self.q_next.save(self.fname+'q_next')
        print('... models saved successfully ...')

    def load_models(self):
        self.q_eval = keras.models.load_model(self.fname+'q_eval')
        self.q_next = keras.models.load_model(self.fname+'q_next')
        print('... models loaded successfully ...')

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                  self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones

    def choose_action(self, observation): #convert to tensor denenecek... expand dims yerine..
        if np.random.random() > self.epsilon:
            observation = np.array(observation, dtype=np.float32)
            #state = np.expand_dims(observation, axis=0)
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.q_eval(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()
        # Sample a batch of experiences from memory
        # states.shape = (32, *input_dims)    -> (32, H, W, C) for images or (32, D) for other state spaces
        # actions.shape = (32,)               -> (32,) because we have one action per state in the batch
        # rewards.shape = (32,)               -> (32,) because we have one reward per state in the batch
        # states_.shape = (32, *input_dims)   -> Same as `states`, the next state
        # dones.shape = (32,)                 -> (32,), indicating whether the episode ended after the action
        states, actions, rewards, states_, dones = self.sample_memory()
        # indices.shape = (32,)               -> A tensor of indices [0, 1, 2, ..., 31]
        indices = tf.range(self.batch_size, dtype=tf.int32)
        # action_indices.shape = (32, 2)      -> Stack of indices and actions, each row is [batch_index, action_index]
        # Example: [[0, action0], [1, action1], ..., [31, action31]]
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            # q_eval(states).shape = (32, 6)   -> Network output, Q-values for all actions in each state
            # q_pred.shape = (32,)             -> Gather Q-values for the specific actions taken, 1 value per state
            q_pred = tf.gather_nd(self.q_eval(states), indices=action_indices)
            # q_next.shape = (32, 6)           -> Network output for next states, Q-values for all actions
            q_next = self.q_next(states_)
            q_eval = self.q_eval(states_)
            #next state için hem q_next hem q_eval hesapladık
            #target seçerken target network max aksiyonlarıını q evalden aldık
            
            max_actions = tf.math.argmax(q_eval, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)
            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx)* \
                    (1 - dones.numpy())
            
            # loss.shape = ()                  -> The loss is scalar, as it's the average MSE over the batch
            loss = keras.losses.MSE(q_pred, q_target)
        # params is a list of variables (weights) in the network
        # grads.shape matches params.shape    -> Gradients have the same shape as the network parameters
        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)
        # Apply gradients to update the network weights
        self.q_eval.optimizer.apply_gradients(zip(grads, params))

        self.learn_step_counter += 1

        self.decrement_epsilon()
        


"""

    
    def learn(self):
        if self.memory_mem_cntr < self.batch_size:
            return
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        
        indices = tf.range(self.batch_size)
        action_indices = tf.stack([indices, actions])
        with tf.gradieant as tape:
            q_pred = tf.gather_nd(q_eval(state), indices=action_indices)
            #p eval state 32, 6 idi, action indislerine göre 32, döndürdüm
            q_next = self.q_next(state_)
            max_actions = tf.argmax(q_next, axis=1)
            max_action_inx = tf.stack(indices, max_actions)
            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices = max_action_idx)*
                (1-dones)
            loss = keras.losses.MSE(q_pred..)

"""


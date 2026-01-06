import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

def DeepQNetwork(input_dims, n_actions):
    """
    SIMPLE DQN network - no dropouts, smaller architecture for faster learning
    """
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=input_dims),
        
        # Single hidden layer
        Dense(64, activation='relu'),
        
        # Output layer - Q-values for each action
        Dense(n_actions, activation=None)  # Linear output for Q-values
    ])
    
    return model
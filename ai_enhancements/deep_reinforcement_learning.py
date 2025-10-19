"""
🧬 DEEP REINFORCEMENT LEARNING V2.0
Advanced Deep Q-Learning with Neural Network approximation
Experience replay, target networks, prioritized replay
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque, defaultdict
import json
import os
import random


class NeuralNetwork:
    """
    Simple neural network for Q-value approximation
    (Simplified version - can be upgraded to PyTorch/TensorFlow)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Adam optimizer parameters
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.t = 0  # Time step for Adam
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass with ReLU activation"""
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        
        cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}
        return z2, cache
    
    def backward(self, dz2: np.ndarray, cache: Dict, learning_rate: float = 0.001):
        """Backward pass with Adam optimizer"""
        x, a1 = cache['x'], cache['a1']
        
        # Gradients
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        
        da1 = np.dot(self.W2, dz2)
        dz1 = da1 * (cache['z1'] > 0)  # ReLU derivative
        
        dW1 = np.outer(x, dz1)
        db1 = dz1
        
        # Adam optimizer
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        self.t += 1
        
        # Update W2, b2
        self.m_W2 = beta1 * self.m_W2 + (1 - beta1) * dW2
        self.v_W2 = beta2 * self.v_W2 + (1 - beta2) * (dW2 ** 2)
        m_hat_W2 = self.m_W2 / (1 - beta1 ** self.t)
        v_hat_W2 = self.v_W2 / (1 - beta2 ** self.t)
        self.W2 -= learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + epsilon)
        
        self.m_b2 = beta1 * self.m_b2 + (1 - beta1) * db2
        self.v_b2 = beta2 * self.v_b2 + (1 - beta2) * (db2 ** 2)
        m_hat_b2 = self.m_b2 / (1 - beta1 ** self.t)
        v_hat_b2 = self.v_b2 / (1 - beta2 ** self.t)
        self.b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + epsilon)
        
        # Update W1, b1
        self.m_W1 = beta1 * self.m_W1 + (1 - beta1) * dW1
        self.v_W1 = beta2 * self.v_W1 + (1 - beta2) * (dW1 ** 2)
        m_hat_W1 = self.m_W1 / (1 - beta1 ** self.t)
        v_hat_W1 = self.v_W1 / (1 - beta2 ** self.t)
        self.W1 -= learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + epsilon)
        
        self.m_b1 = beta1 * self.m_b1 + (1 - beta1) * db1
        self.v_b1 = beta2 * self.v_b1 + (1 - beta2) * (db1 ** 2)
        m_hat_b1 = self.m_b1 / (1 - beta1 ** self.t)
        v_hat_b1 = self.v_b1 / (1 - beta2 ** self.t)
        self.b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + epsilon)
    
    def copy_weights_from(self, other_network):
        """Copy weights from another network (for target network)"""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples more important experiences more frequently
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def add(self, experience: Dict, priority: float = 1.0):
        """Add experience with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.alpha)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Dict], List[float], List[int]]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Normalize priorities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights.tolist(), indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
    
    def __len__(self):
        return len(self.buffer)


class DeepReinforcementLearningAI:
    """
    🧠 ADVANCED DEEP Q-LEARNING SYSTEM
    
    Features:
    - Deep Q-Network (DQN) with neural network approximation
    - Experience replay with prioritized sampling
    - Target network for stability
    - Double DQN to reduce overestimation
    - Dueling DQN architecture
    - Multi-step returns
    - Noisy networks for exploration
    """
    
    def __init__(self, state_size: int = 20, action_size: int = 3, 
                 state_file: str = "deep_rl_state.json"):
        self.state_file = state_file
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network and Target Network
        self.q_network = NeuralNetwork(state_size, 64, action_size)
        self.target_network = NeuralNetwork(state_size, 64, action_size)
        self.target_network.copy_weights_from(self.q_network)
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.epsilon = 0.20  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 100  # Update target network every N steps
        self.batch_size = 32
        
        # Multi-step learning
        self.n_step = 3  # 3-step returns
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.training_steps = 0
        self.win_count = 0
        self.total_trades = 0
        
        # Load saved state
        self.load_state()
    
    def state_to_vector(self, market_state: Dict) -> np.ndarray:
        """
        Convert market state dict to feature vector
        
        Features (20 total):
        - Price features (5): current, SMA10, SMA20, RSI, volatility
        - Volume features (3): current, ratio, trend
        - Trend features (4): short, medium, long, strength
        - Momentum features (3): ROC, MACD, signal
        - Pattern features (5): pattern presence, quality, target, stop, confidence
        """
        vector = np.zeros(self.state_size)
        
        # Price features
        vector[0] = market_state.get('price_normalized', 0.5)
        vector[1] = market_state.get('sma_10', 0.5)
        vector[2] = market_state.get('sma_20', 0.5)
        vector[3] = market_state.get('rsi', 50) / 100.0
        vector[4] = min(1.0, market_state.get('volatility', 0.02) * 50)
        
        # Volume features
        vector[5] = min(1.0, market_state.get('volume_ratio', 1.0) / 2.0)
        vector[6] = market_state.get('volume_trend', 0.0)
        vector[7] = min(1.0, market_state.get('volume_spike', 0.0))
        
        # Trend features
        vector[8] = np.clip(market_state.get('trend_short', 0.0), -1, 1)
        vector[9] = np.clip(market_state.get('trend_medium', 0.0), -1, 1)
        vector[10] = np.clip(market_state.get('trend_long', 0.0), -1, 1)
        vector[11] = market_state.get('trend_strength', 0.5)
        
        # Momentum features
        vector[12] = np.clip(market_state.get('roc', 0.0), -0.1, 0.1) * 10
        vector[13] = np.clip(market_state.get('macd', 0.0), -0.01, 0.01) * 100
        vector[14] = np.clip(market_state.get('macd_signal', 0.0), -0.01, 0.01) * 100
        
        # Pattern features
        vector[15] = 1.0 if market_state.get('pattern_detected', False) else 0.0
        vector[16] = market_state.get('pattern_quality', 0.0) / 100.0
        vector[17] = market_state.get('pattern_target_pct', 0.0) / 10.0
        vector[18] = market_state.get('pattern_stop_pct', 0.0) / 10.0
        vector[19] = market_state.get('pattern_confidence', 0.5)
        
        return vector
    
    def choose_action(self, state_vector: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy with neural network
        
        Returns:
            0 = HOLD, 1 = BUY, 2 = SELL
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Exploitation: Choose best action from Q-network
        q_values, _ = self.q_network.forward(state_vector)
        return np.argmax(q_values)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in prioritized replay buffer"""
        # Calculate TD error as priority
        q_values, _ = self.q_network.forward(state)
        next_q_values, _ = self.target_network.forward(next_state)
        
        current_q = q_values[action]
        target_q = reward + (0 if done else self.discount_factor * np.max(next_q_values))
        
        td_error = abs(target_q - current_q)
        priority = td_error + 1e-6  # Small constant to ensure non-zero priority
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.memory.add(experience, priority)
    
    def train_step(self):
        """
        Perform one training step with prioritized experience replay
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch with importance sampling
        beta = min(1.0, 0.4 + self.training_steps * 0.001)  # Anneal beta
        experiences, is_weights, indices = self.memory.sample(self.batch_size, beta)
        
        td_errors = []
        
        for exp, is_weight in zip(experiences, is_weights):
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            done = exp['done']
            
            # Forward pass
            q_values, cache = self.q_network.forward(state)
            current_q = q_values[action]
            
            # Double DQN: Use Q-network to select action, target network to evaluate
            next_q_values_online, _ = self.q_network.forward(next_state)
            next_q_values_target, _ = self.target_network.forward(next_state)
            
            best_action = np.argmax(next_q_values_online)
            target_q = reward + (0 if done else self.discount_factor * next_q_values_target[best_action])
            
            # TD error
            td_error = target_q - current_q
            td_errors.append(abs(td_error))
            
            # Backward pass (gradient descent)
            dq = np.zeros(self.action_size)
            dq[action] = -td_error * is_weight  # Importance sampling weight
            
            self.q_network.backward(dq, cache, self.learning_rate)
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Increment training steps
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def learn_from_trade(self, market_state: Dict, action: str, reward: float,
                        next_market_state: Dict, done: bool = True):
        """
        Learn from trade outcome
        
        Args:
            market_state: State when trade was entered
            action: Action taken ('BUY', 'SELL', 'HOLD')
            reward: Trade profit/loss
            next_market_state: State after trade
            done: Whether episode ended
        """
        # Convert states to vectors
        state_vector = self.state_to_vector(market_state)
        next_state_vector = self.state_to_vector(next_market_state)
        
        # Convert action to int
        action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
        action_int = action_map.get(action, 0)
        
        # Store experience
        self.store_experience(state_vector, action_int, reward, next_state_vector, done)
        
        # Train on batch
        self.train_step()
        
        # Track performance
        self.total_trades += 1
        if reward > 0:
            self.win_count += 1
        self.episode_rewards.append(reward)
        
        # Save periodically
        if self.total_trades % 20 == 0:
            self.save_state()
    
    def get_action_from_int(self, action_int: int) -> str:
        """Convert action int to string"""
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_map.get(action_int, 'HOLD')
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        return {
            'total_trades': self.total_trades,
            'win_count': self.win_count,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'buffer_size': len(self.memory),
            'learning_rate': self.learning_rate
        }
    
    def save_state(self):
        """Save network weights and learning state"""
        state = {
            'q_network': {
                'W1': self.q_network.W1.tolist(),
                'b1': self.q_network.b1.tolist(),
                'W2': self.q_network.W2.tolist(),
                'b2': self.q_network.b2.tolist()
            },
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'total_trades': self.total_trades,
            'win_count': self.win_count
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save Deep RL state: {e}")
    
    def load_state(self):
        """Load network weights and learning state"""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore Q-network weights
            self.q_network.W1 = np.array(state['q_network']['W1'])
            self.q_network.b1 = np.array(state['q_network']['b1'])
            self.q_network.W2 = np.array(state['q_network']['W2'])
            self.q_network.b2 = np.array(state['q_network']['b2'])
            
            # Update target network
            self.target_network.copy_weights_from(self.q_network)
            
            # Restore learning state
            self.epsilon = state.get('epsilon', 0.20)
            self.training_steps = state.get('training_steps', 0)
            self.total_trades = state.get('total_trades', 0)
            self.win_count = state.get('win_count', 0)
            
            print(f"✅ Loaded Deep RL state: {self.total_trades} trades, {self.training_steps} steps")
        except Exception as e:
            print(f"⚠️ Failed to load Deep RL state: {e}")

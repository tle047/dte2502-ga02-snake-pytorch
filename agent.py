"""
store all the agents here - PyTorch Version
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

def huber_loss(y_true, y_pred, delta=1):
    """PyTorch implementation for huber loss
    loss = {
        0.5 * (y_true - y_pred)**2 if abs(y_true - y_pred) < delta
        delta * (abs(y_true - y_pred) - 0.5 * delta) otherwise
    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        loss values for all points
    """
    error = y_true - y_pred
    quad_error = 0.5 * torch.square(error)
    lin_error = delta * (torch.abs(error) - 0.5 * delta)
    # quadratic error, linear error
    return torch.where(torch.abs(error) < delta, quad_error, lin_error)

def mean_huber_loss(y_true, y_pred, delta=1):
    """Calculates the mean value of huber loss

    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        average loss across points
    """
    return torch.mean(huber_loss(y_true, y_pred, delta))

class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        os.makedirs(file_path, exist_ok=True)
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col


class DQNNetwork(nn.Module):
    """PyTorch CNN Network for Deep Q-Learning
    
    Architecture matches v17.1 config:
    - Conv2D: 16 filters, (3,3) kernel, padding='same', ReLU
    - Conv2D: 32 filters, (3,3) kernel, ReLU
    - Conv2D: 64 filters, (5,5) kernel, ReLU
    - Flatten
    - Dense: 64 units, ReLU
    - Output: n_actions units, Linear
    """
    def __init__(self, board_size, n_frames, n_actions):
        super(DQNNetwork, self).__init__()
        
        # Conv layers
        # Input: (batch, channels, height, width) = (N, n_frames, board_size, board_size)
        self.conv1 = nn.Conv2d(n_frames, 16, kernel_size=3, padding=1)  # 'same' padding
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        
        # Calculate flattened size after convolutions
        # After conv1 (padding=1): 10x10 -> 10x10
        # After conv2 (no padding, kernel=3): 10x10 -> 8x8
        # After conv3 (no padding, kernel=5): 8x8 -> 4x4
        # So final size: 64 channels * 4 * 4 = 1024
        self.flatten_size = 64 * 4 * 4
        
        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten - use reshape instead of view
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Linear activation (no activation function)
        return x


class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning - PyTorch Version
    model outputs everywhere refers to Q values
    This class extends to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _model : PyTorch Module
        Stores the DQN model
    _target_net : PyTorch Module
        Stores the target network of the DQN model
    device : torch.device
        Device to run the model on (CPU/GPU)
    optimizer : torch.optim.Optimizer
        Optimizer for training the model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.reset_models()

    def reset_models(self):
        """ Reset all the models by creating new networks"""
        # main network
        self._model = self._agent_model()
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)

        # target network
        if self._use_target_net:
            self._target_net = self._agent_model()
            self.update_target_net()

    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : PyTorch Module
            DQN model
        """
        # Load config from JSON file
        config_path = 'model_config/{:s}.json'.format(self._version)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                m = json.loads(f.read())
        else:
            # Default config if file not found
            m = {
                'board_size': self._board_size,
                'frames': self._n_frames,
                'n_actions': self._n_actions
            }
        
        # Create model
        model = DQNNetwork(self._board_size, self._n_frames, self._n_actions)
        model = model.to(self.device)
        
        
        return model

    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        return board.copy()

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : PyTorch Module, optional
            The model to use for prediction, model or target network

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        
        # Convert numpy to PyTorch tensor
        # Input is NHWC format (batch, height, width, channels)
        # Need to convert to NCHW format (batch, channels, height, width)
        board_tensor = torch.FloatTensor(board).to(self.device)
        board_tensor = board_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Set model to evaluation mode and disable gradients
        model.eval()
        with torch.no_grad():
            model_outputs = model(board_tensor)
            model_outputs = model_outputs.cpu().numpy()  # Convert back to numpy
        
        return model_outputs

    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        return board.astype(np.float32)/4.0

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves==1, model_outputs, -np.inf), axis=1)

    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using PyTorch's save function
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        
        os.makedirs(file_path, exist_ok=True)
        
        # Save model state dict and optimizer state
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration
        }, "{}/model_{:04d}.pth".format(file_path, iteration))
        
        if(self._use_target_net):
            torch.save({
                'model_state_dict': self._target_net.state_dict(),
                'iteration': iteration
            }, "{}/model_{:04d}_target.pth".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """Load models from disk using PyTorch's load function
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        
        try:
            # Load model
            checkpoint = torch.load("{}/model_{:04d}.pth".format(file_path, iteration),
                                   map_location=self.device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load target network if used
            if(self._use_target_net):
                checkpoint_target = torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration),
                                              map_location=self.device)
                self._target_net.load_state_dict(checkpoint_target['model_state_dict'])
        except FileNotFoundError:
            print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model)
        print(f'\nTotal parameters: {sum(p.numel() for p in self._model.parameters())}')
        print(f'Trainable parameters: {sum(p.numel() for p in self._model.parameters() if p.requires_grad)}')
        if(self._use_target_net):
            print('\nTarget Network')
            print(self._target_net)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward + 
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        # Sample from replay buffer
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        
        if(reward_clip):
            r = np.sign(r)
        
        # Convert to PyTorch tensors
        s_tensor = torch.FloatTensor(self._normalize_board(s)).to(self.device)
        a_tensor = torch.FloatTensor(a).to(self.device)  # One-hot encoded actions
        r_tensor = torch.FloatTensor(r).to(self.device)
        next_s_tensor = torch.FloatTensor(self._normalize_board(next_s)).to(self.device)
        done_tensor = torch.FloatTensor(done).to(self.device)
        legal_moves_tensor = torch.FloatTensor(legal_moves).to(self.device)
        
        # Convert NHWC to NCHW format
        s_tensor = s_tensor.permute(0, 3, 1, 2)
        next_s_tensor = next_s_tensor.permute(0, 3, 1, 2)
        
        # Get next state Q values from target network
        current_model = self._target_net if self._use_target_net else self._model
        current_model.eval()
        with torch.no_grad():
            next_q_values = current_model(next_s_tensor)
            
            # Mask illegal moves with -inf
            next_q_values_masked = torch.where(legal_moves_tensor == 1, 
                                               next_q_values, 
                                               torch.tensor(-float('inf')).to(self.device))
            max_next_q = torch.max(next_q_values_masked, dim=1)[0].unsqueeze(1)
        
        # Calculate target Q values: r + gamma * max(Q(s', a')) * (1 - done)
        discounted_reward = r_tensor + (self._gamma * max_next_q) * (1 - done_tensor)
        
        # Get current Q values
        self._model.train()
        current_q_values = self._model(s_tensor)
        
        # Create target: update only the taken actions
        # a_tensor is one-hot encoded, so we use it to select which Q-values to update
        target = current_q_values.clone().detach()
        target = (1 - a_tensor) * target + a_tensor * discounted_reward
        
        # Compute loss (Huber loss)
        loss = mean_huber_loss(target, current_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if(self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        """Simple utility function to check if the model and target 
        network have the same weights or not
        """
        model_state = self._model.state_dict()
        target_state = self._target_net.state_dict()
        
        for key in model_state:
            if key in target_state:
                match = torch.equal(model_state[key], target_state[key])
                print(f'Layer {key} Match: {match.item() if match.numel() == 1 else match.all().item()}')


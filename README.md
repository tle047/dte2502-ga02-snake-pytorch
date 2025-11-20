# Snake RL - PyTorch Deep Q-Learning Agent

## Graded Assignment 02

This project is my solution for DTE2502 – Neural Networks, Graded Assignment 02.

For the final model I use **version v17.1** with:
- board size **10×10**
- random **obstacles**
- **2 input frames**
- **4 actions** (up, down, left, right)

## Running Graded Assignment 02 (local / GA01 env)

1. Create/activate the GA01 conda environment (same as in GA01).

2. Clone this repository:  
   `git clone https://github.com/tle047/dte2502-ga02-snake-pytorch.git`  
   `cd dte2502-ga02-snake-pytorch`

3. Run training:  
   `python training.py`

4. The final model will be saved to:  
   `models/v17.1/model_200000.pth`  
   Training logs will be saved to:  
   `model_logs/v17.1.csv`

---

## Quick Start for Google Colab

The Colab workflow is driven by the notebook **`updatedmain.ipynb`**.

### Step 1: Mount Drive and (optionally) unzip a package

Mount Google Drive:  
`from google.colab import drive`  
`drive.mount('/content/drive')`

If you have a prepared zip (for example `colab_package.zip` in `/content`), unzip it, otherwise you can just clone the GitHub repo directly in Colab.

### Step 2: Install dependencies and move into the project

Install basic dependencies:  
`!pip install torch numpy matplotlib pandas tqdm`

Change directory into the project (for example after cloning or unzipping):  
`%cd /content/dte2502-ga02-snake-pytorch`

### Step 3: Generate obstacle boards and run training

Generate obstacle boards for v17.1:  
`!python obstacles_board_generator.py`

Run training:  
`!python training.py`

Training uses **version v17.1** with a 10×10 board, random obstacles, 2 frames and 4 actions.  
Models and logs are written to:

- `models/v17.1/`  
- `model_logs/v17.1.csv`

### Step 4: Backup logs and models to Google Drive

In `updatedmain.ipynb` the following are backed up to a Drive folder (for example `GA02_Snake_DQN_backup`):

- `model_logs/v17.1.csv`  
- `models/v17.1/`

Target path (example):  
`/content/drive/MyDrive/GA02_Snake_DQN_backup/`

### Step 5: Plot training curves

The notebook loads `model_logs/v17.1.csv` and creates three plots:

- training reward vs iteration (raw + moving average)  
- episode length vs iteration (raw + moving average)  
- training loss vs iteration  

Each plot is saved locally in the project folder and also copied to the Drive backup folder.

### Step 6: Evaluate the trained agent

`updatedmain.ipynb` then:

- creates a `SnakeNumpy` environment with board size 10, 2 frames and 4 actions  
- creates a `DeepQLearningAgent` with `version='v17.1'`  
- loads the checkpoint `models/v17.1/model_200000.pth`  
- runs 50 evaluation games using a **greedy policy** (`epsilon = 0.0`) via `play_game2`

The notebook prints:

`Eval avg reward: 232`  
`Eval avg length: 260`

These are the numbers reported in the **Results** section below.

### Step 7: Visualize one game

Finally, the notebook:

- plays one game with the trained agent on an obstacle board  
- records snapshots of the board  
- visualizes them in a 1×5 subplot figure using a colormap where:

  - light green = snake head  
  - darker green = snake body  
  - dark gray = obstacles  
  - red = food  

This qualitatively matches the behavior shown in the original TensorFlow Snake RL project.

---

## Files Included

- `agent.py` – PyTorch `DeepQLearningAgent`  
- `training.py` – Training script (200k episodes for v17.1)  
- `game_environment.py` – Snake game environment  
- `replay_buffer.py` – Experience replay buffer  
- `utils.py` – Utility functions (playing games, epsilon-greedy, etc.)  
- `obstacles_board_generator.py` – Obstacle board generator for v17.1  
- `model_config/v17.1.json` – Model configuration  
- `model_logs/v17.1.csv` – Training log for v17.1  
- `updatedmain.ipynb` – Colab notebook for training, evaluation, plotting and packaging  

---

## Model Configuration (v17.1)

- board size: **10×10**  
- frames: **2**  
- actions: **4**  
- replay buffer size: **80,000**  
- batch size: **64**  
- episodes: **200,000**  
- discount factor: **γ = 0.99**  
- optimizer: **RMSprop**, learning rate `0.0005`  
- target network: enabled (`use_target_net = True`)  

---

## Expected Output

- trained model checkpoint: `models/v17.1/model_200000.pth`  
- training logs: `model_logs/v17.1.csv`  
- training plots (saved by the notebook): reward, episode length and loss curves  
- one-game visualization figure of the trained policy on an obstacle board  

---

## Results (v17.1)

A greedy evaluation (`epsilon = 0.0`) over **50 games** on obstacle boards gives:

- average reward ≈ **232**  
- average episode length ≈ **260** steps  

The trained agent learns to move around obstacles, avoid walls and reliably reach the food.

---

## Notes

- The script will use **GPU** if available.  
- Training logs and checkpoints are written every **500 episodes**.  
- The final model used in the assignment is `model_200000.pth` in `models/v17.1/`.  

---

## Acknowledgements

This project is based on the TensorFlow Snake RL repository provided by **Kalyan** for the DTE2502 course:

- Original TF code: https://github.com/DragonWarrior15/snake-rl  

In this assignment I re-implemented the `DeepQLearningAgent` and the training pipeline in **PyTorch**, while keeping the overall environment, board configuration and visualization structure inspired by the original project.

GitHub repository for this PyTorch version:  
`https://github.com/tle047/dte2502-ga02-snake-pytorch`

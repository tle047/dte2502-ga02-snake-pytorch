# Snake RL - PyTorch Deep Q-Learning Agent

## Graded Assignment 02

This project is my solution for DTE2502 – Neural Networks, Graded Assignment 02.

For the final model I use **version v17.1** with:
- board size **10×10**
- random **obstacles**
- **2 input frames**
- **4 actions** (up, down, left, right)

## Running Graded Assignment 02

1. Create/activate the GA01 conda environment (same as in GA01).
2. Clone this repository:
   git clone https://github.com/tle047/dte2502-ga02-snake-pytorch.git
   cd dte2502-ga02-snake-pytorch
3. Run training:
   python training.py
4. The final model will be saved to:
   models/v17.1/model_200000.pth

## Quick Start for Google Colab

### Step 1: Upload this zip file to Colab
1. On GitHub, click **Code → Download ZIP** to download this repository.
2. Upload the downloaded ZIP file (`dte2502-ga02-snake-pytorch-main.zip`) to Colab.
3. Extract:
   ```python
   !unzip dte2502-ga02-snake-pytorch-main.zip
   %cd dte2502-ga02-snake-pytorch-main
   ```

### Step 2: Enable GPU
- Runtime → Change runtime type → GPU

### Step 3: Install Dependencies
```python
!pip install torch torchvision numpy pandas matplotlib tqdm
```

### Step 4: Run Training
```python
!python training.py
```

Training time depends on the GPU. On Colab GPU it took around 30–40 minutes.
Model checkpoints are saved every 500 episodes.

### Step 5: Save to Google Drive
```python
!mkdir -p /content/drive/MyDrive/GA02_model/v17.1
!cp models/v17.1/model_200000.pth /content/drive/MyDrive/GA02_model/v17.1/
!cp models/v17.1/obstacles_board /content/drive/MyDrive/GA02_model/v17.1/
```
This copies the final model and the obstacle boards for version v17.1 to Google Drive.


## Files Included

- `agent.py` - PyTorch DeepQLearningAgent
- `training.py` - Training script (200k episodes)
- `game_environment.py` - Snake game environment
- `replay_buffer.py` - Experience replay buffer
- `utils.py` - Utility functions
- `obstacles_board_generator.py` - Obstacle board generator
- `model_config/v17.1.json` - Model configuration

## Model Configuration

- Board size: 10x10
- Frames: 2
- Actions: 4
- Buffer size: 80,000
- Batch size: 64
- Episodes: 200,000

## Expected Output

- Trained model: `models/v17.1/model_200000.pth`
- Training logs: `model_logs/v17.1.csv`

### Results (v17.1)

A greedy evaluation (epsilon = 0.0) over 50 games on obstacle boards gives:

- average reward ≈ **232**
- average episode length ≈ **260** steps

## Notes

- The script will use GPU if available.
- Training logs and checkpoints are written every 500 episodes.
- The final model used in the assignment is `model_200000.pth`.



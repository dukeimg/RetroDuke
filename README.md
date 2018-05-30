# RetroDuke

Neural networks play my favorite retro games

<img src="https://media.giphy.com/media/AhhTXa52vdkP4p0N8W/giphy.gif" />

## Featured games:

- Top Gear 2
- Contra: Hard Corps

## Featured networks:

- Deep Q Neural Network
- Monte Carlo Policy Gradients

## Usage:

1. Create and activate a Python 3.6 virtual environment in `venv` directory.
2. Install required python packages: `pip install -r requirements.txt`
3. Run `copy.sh` to inject my favourite games environments to [Retro](https://github.com/openai/retro)

### train.py

Trains new agent

```
python train.py <game> <network> --optional-parameter <optional_parameter_value>
```

Required parameters: 

- `game`. List of available games (with exact spelling) can be found in `game_envs` directory
- `network`. Can be either `dqn` or `pgn`

Optional parameters:

```
python train.py <game> <network> --optional-parameter <optional_parameter_value>
```

- `--state`. The initial state file to load, minus the extension. States allows you to run game on its different moments.
- `--model`, `-m`. The model to load, minus the extension. Use this parameter to resume your training.
- `--scenario`, `-s`. The scenario file to load, minus the extension.
- `--quiet`, `-q`. Do not render gameplay.

### play.py

Playing one game session using an existing agent

```
python play.py <game> <network> --optional-parameter <optional_parameter_value>
```

Required parameters: 

- `game`. List of available games (with exact spelling) can be found in `game_envs` directory
- `network`. Can be either `dqn` or `pgn`

Optional parameters:

```
python play.py <game> <network> --optional-parameter <optional_parameter_value>
```

- `--state`. The initial state file to load, minus the extension. States allows you to run game on its different moments.
- `--model`, `-m`. The model to load, minus the extension. Use this parameter to resume your training. By default the 
last model is used.

## Upcoming games:

- Doom Troopers
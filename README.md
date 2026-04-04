# AI Debate System

A system for running automated debates between AI agents with a judge to evaluate the arguments.

## Features

- Two AI agents debate any given topic (pro and con positions)
- A judge agent evaluates the arguments and provides a final verdict
- Configurable parameters for creativity and strictness
- Token usage tracking
- Dynamic debate length based on judge evaluation

## Installation

1. Ensure you have Ollama installed and running with the "my-gemma" model available
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command Line Version

Run the main debate script:
```
python debate_agents_lc.py
```

*Note: Topics and debate parameters (such as creativity levels and max rounds) are configured directly in the code at the bottom of the script.*

You can also run a simpler version of the debate logic:
```
python debate_agents_simple.py
```

### Web UI Version

A web-based UI is available in the `ui` directory:

1. Navigate to the ui directory:
   ```
   cd ui
   ```

2. Install UI dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run debate_ui.py
   ```

Alternatively, from the main directory:
```
python run_ui.py
```

The UI allows you to:
- Enter custom debate topics
- Adjust debate parameters (number of rounds, creativity levels)
- View the debate conversation in a chat-like interface
- See the judge's final verdict and statistics
- Watch the debate unfold in real-time with live updates

## Configuration

Key parameters that can be adjusted:
- `max_rounds`: Maximum number of debate rounds (default: 3)
- `pro_temp`: Proponent creativity level (0.0-1.0, default: 0.9)
- `con_temp`: Opponent creativity level (0.0-1.0, default: 0.7)
- `judge_temp`: Judge strictness (0.0-1.0, default: 0.3)

## Requirements

- Python 3.7+
- Ollama with the "my-gemma" model
- Langchain-core
- Streamlit (for UI version)
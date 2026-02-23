# AI Debate UI

A simple web interface for watching AI agents debate topics with a judge.

## Features

- Text-based conversation view showing arguments from both sides
- Special styling for judge messages
- Configurable debate parameters (number of rounds, creativity levels)
- Topic input field for custom debates

## Installation

1. Navigate to the ui directory:
   ```
   cd ui
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run debate_ui.py
```

The app will open in your browser. Enter a debate topic and click "Start Debate" to begin.

## Configuration

Adjust these parameters in the sidebar:
- Maximum Rounds: How many rounds of debate to allow (1-10)
- Proponent Creativity: Higher values make the pro side more creative (0.0-1.0)
- Opponent Creativity: Higher values make the con side more creative (0.0-1.0)
- Judge Strictness: Higher values make the judge more strict (0.0-1.0)
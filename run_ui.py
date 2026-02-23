#!/usr/bin/env python3
"""
Script to run the AI Debate UI.
"""

import os
import subprocess
import sys


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the UI script
    ui_script = os.path.join(script_dir, "ui", "debate_ui.py")

    # Check if the UI script exists
    if not os.path.exists(ui_script):
        print(f"Error: UI script not found at {ui_script}")
        sys.exit(1)

    # Run the Streamlit app
    try:
        subprocess.run(["streamlit", "run", ui_script], check=True)
    except FileNotFoundError:
        print(
            "Error: Streamlit not found. Please install it with 'pip install streamlit'"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nUI stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()

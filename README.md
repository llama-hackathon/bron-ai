# Step 1: Delete existing virtualenv
rm -rf /Users/james/Documents/Projects/bron-ai/bron-ai-env

# Step 2: Recreate the venv
python3.11 -m venv /Users/james/Documents/Projects/bron-ai/bron-ai-env

# Step 3: Activate it
source /Users/james/Documents/Projects/bron-ai/bron-ai-env/bin/activate

# Step 4: Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

ffmpeg -i game_2.mp4 -t 01:00:00 -c copy game_2_60.mp4
Development bench of application for controlling mouse cursor using eye/face movement.


Setting up development environment:

- install python3 
- `sudo pip3 install --upgrade pip setuptools wheel`
- `pip3 install -r .requirements.txt`


Application pipeline:

Step 0: Capturing video cam data for training

`python3 -m pipeline2.step0_raw_data_live_cam` 

Step 1: Obtaining face/eye landmarks based on the data captured

Step 2: Generating features for training machine learning models

Step 3: Training and executing models

Step 4: Running live cam simulation based on trained model
This is the code and videos for Project 6 Visual Navigation.

Videos:

https://drive.google.com/drive/folders/120JUYTih8CazfUHH9Z2beVKomUJhsWXb

- The pedestrian detection video  

- The visualization of SCAND dataset

- The visualization of Pedestrian related depth data (todo)


Code:

- The code for pedestrian detection (see `pedestrian_detect/`)

- The code for ROSBag parsing (mainly for depth images) (see `parse_rosbag.py`)

- The code for BEV transformation (see `depth2ray.py`) (todo: producing video)

To run:

```bash
./docker_run.sh
```

Then you can run the parsing code in the docker container.
```bash
python3 extract_rosbag.py
```


import sys
from pathlib import Path
import json
import os
# python3 detect.py --source /Users/shamsansari/Desktop/UAV-Berkeley/suas_2022/images/san_fransico.jpg --save-txt --save-crop --save-conf:q
# python3 data_parser.py 2 0.520686 0.597917 0.0579196 0.0708333 0.313508

def save(cls, x, y, w, h, conf=''):
    dir_path = "output"
    file_name = "test.json"
    file_path = os.path.join(dir_path, file_name)

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    data = {"class": cls,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf}

    with open(file_path, "w") as file:
        file.write(json.dumps(data))
    print(f"Created file in {file_path}")
    

if __name__ == "__main__":
    save(*sys.argv[1:])


# args :
# class, x, y, w, h, conf
#  Confidence is optional

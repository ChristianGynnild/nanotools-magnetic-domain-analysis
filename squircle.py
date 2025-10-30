from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class Squircle:
    figure_name: str
    index: int
    squircle_factor: int
    data: np.ndarray

    def serialize(self, filename:str):
        squircle_dict = asdict(self)
        squircle_data_json = json.dumps({key:squircle_dict[key] for key in squircle_dict if (key!="data")})

        print("json data:", squircle_data_json)

        filename_path = Path(filename)
        with open(str(filename_path.with_suffix(".json")), "w") as file:
            file.write(squircle_data_json)
        
        with open(str(filename_path.with_suffix(".npy")), "wb") as file:
            np.save(file, self.data, allow_pickle=False)

def deserialize(filename):
    filename_path = Path(filename)
    with open(str(filename_path.with_suffix(".json")), "r") as file:
        squircle_data_json = file.read()
    
    with open(str(filename_path.with_suffix(".npy")), "rb") as file:
        data = np.load(file, allow_pickle=False)
    
    squircle_data_dict = json.loads(squircle_data_json)
    squircle = Squircle(**(squircle_data_dict | {"data":data}))

    return squircle
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import json

def lerp(a,b,x):
    return a*(1-x) + b*(x)

size_of_picture = 40


@dataclass
class Squircle:
    figure_name: str
    index: int
    squircle_factor: int
    data: np.ndarray
    pixel_size:float
    width_nm:int

    def serialize(self, filename:str):
        squircle_dict = asdict(self)
        squircle_data_json = json.dumps({key:squircle_dict[key] for key in squircle_dict if (key!="data")})

        filename_path = Path(filename)
        with open(str(filename_path.with_suffix(".json")), "w") as file:
            file.write(squircle_data_json)
        
        with open(str(filename_path.with_suffix(".npy")), "wb") as file:
            np.save(file, self.data, allow_pickle=False)

    @property
    def width_pixels(self):
        return self.width_nm/self.pixel_size

    def get_squircle_mask(self):
        """
        Produces an ellipse. Radius changes with N as radius =  N+7 and rotation as rotation=k*10 degrees
        """
        # Count is number of figures from square to circle
        size = self.width_pixels
        radius = lerp(0, size, (self.squircle_factor)/(5))
        x = np.arange(-int(size_of_picture/2),int(size_of_picture/2))
        y = np.arange(-int(size_of_picture/2),int(size_of_picture/2))
        xx,yy = np.meshgrid(x,y,indexing='ij')

        cross_width = size - radius
        circle_x_position = abs(size) - radius
        circle_y_position = abs(size) - radius

        pixels = np.logical_or(
            np.logical_and((xx**2) <= size**2, (yy**2) <= cross_width**2),
            np.logical_or(
                np.logical_and((yy**2) <= size**2, (xx**2) <= cross_width**2),
                (np.abs(xx)-circle_x_position)**2 + (np.abs(yy) - circle_y_position)**2 <= radius**2
            )
        )
        return pixels
    
    def get_masked_data(self):
        pass


def deserialize(filename):
    filename_path = Path(filename)
    with open(str(filename_path.with_suffix(".json")), "r") as file:
        squircle_data_json = file.read()
    
    with open(str(filename_path.with_suffix(".npy")), "rb") as file:
        data = np.load(file, allow_pickle=False)
    
    squircle_data_dict = json.loads(squircle_data_json)
    squircle = Squircle(**(squircle_data_dict | {"data":data}))

    return squircle
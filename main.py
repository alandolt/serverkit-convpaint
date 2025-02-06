from typing import List, Literal, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit
from napari_convpaint import conv_paint, conv_paint_utils
import os

custom_model_path = "/models"
models_list = [(f.name).removesuffix(".pickle") for f in os.scandir(custom_model_path)]
print(f"Available models: {models_list}")
default_model = str(models_list[0])
models_list = Literal[tuple(models_list)]


class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D).",
        json_schema_extra={"widget_type": "image"},
    )
    model_name: models_list = Field(
        default=default_model,
        title="Model",
        description="The model used for semantic segmentation",
        json_schema_extra={"widget_type": "dropdown"},
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim != 2:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array

# Define the run_algorithm() method for your algorithm
class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str="convpaint",
        parameters_model: Type[BaseModel]=Parameters
    ):
        super().__init__(algorithm_name, parameters_model)
        self.last_model = None
        self.last_model_name = None
        self.last_random_forest = None
        self.last_model_param = None
        self.last_model_state = None

    def run_algorithm(
        self,
        image: np.ndarray,
        model_name: str,
        **kwargs,
    ) -> List[tuple]:
        """Runs the algorithm."""
        print("Running algorithm")
        if model_name != self.last_model_name:
            random_forest, model, model_param, model_state = conv_paint.load_model(
                os.path.join(custom_model_path, model_name + ".pickle")
            )
            model_param.fe_use_cuda = True
            self.last_model = model
            self.last_model_name = model_name
            self.last_random_forest = random_forest
            self.last_model_param = model_param
            self.last_model_state = model_state
        else: 
            model = self.last_model
            random_forest = self.last_random_forest
            model_param = self.last_model_param
            model_state = self.last_model_state
        
        print(type(image))
        print("Predicting image")
        mean, std = conv_paint_utils.compute_image_stats(image)
        print(mean, std)
        print("Normalizing image")
        img_normed = conv_paint_utils.normalize_image(image, mean, std)
        print("Predicting image")
        segmentation = model.predict_image(
            img_normed, random_forest, model_param
        )
        print("segmentation finished")
        segmentation_params = {"name": "Convpaint result"}
       
        return [
            (segmentation, segmentation_params, "labels3d"),
        ]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Load one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images

server = Server()
app = server.app

if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
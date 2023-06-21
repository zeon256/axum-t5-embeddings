from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from InstructorEmbedding import INSTRUCTOR

app = FastAPI()
model = INSTRUCTOR('path')

class Item(BaseModel):
    inputs: list[str]

@app.post("/feature-extraction")
def create_item(item: Item):
    embeddings = model.encode(item.inputs, show_progress_bar=True)
    embeddings_array = np.array(embeddings)
    return embeddings_array.tolist()

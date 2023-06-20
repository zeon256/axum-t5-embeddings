from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from InstructorEmbedding import INSTRUCTOR

app = FastAPI()
model = INSTRUCTOR('/home/zeon256/Documents/work/copilot/text-generation-webui2/models/hkunlp_instructor-large/')

class Item(BaseModel):
    inputs: list[str]

@app.post("/feature-extraction")
def create_item(item: Item):
    embeddings = model.encode([item.inputs], show_progress_bar=True)
    embeddings_array = np.array(embeddings)
    return embeddings_array.tolist()

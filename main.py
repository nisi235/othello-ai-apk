from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import os

app = FastAPI()

# 静的ファイル（HTML, JS, CSSなど）を提供
app.mount("/static", StaticFiles(directory="static"), name="static")

# トップページで index.html を返す
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ONNXモデルのパス
model_paths = {
    "ai1": "models/othello_ai1.onnx",
    "ai2": "models/othello_ai2.onnx",
    "ai3": "models/othello_ai3.onnx",
    "ai4": "models/othello_ai4.onnx",
    "ai5": "models/othello_ai5.onnx",
}

# セッション初期化
sessions = {}
for key, path in model_paths.items():
    if os.path.exists(path):
        sessions[key] = ort.InferenceSession(path)
    else:
        print(f"モデルファイルが見つかりません: {path}")

# リクエストボディのスキーマ
class BoardRequest(BaseModel):
    model_key: str  # ai1 ~ ai5
    board: list[list[int]]  # 8x8の盤面

# AIによる次の手予測API
@app.post("/predict")
async def predict_next_move(req: BoardRequest):
    if req.model_key not in sessions:
        raise HTTPException(status_code=400, detail="Invalid model_key")

    sess = sessions[req.model_key]
    board_array = np.array(req.board, dtype=np.float32)
    
    if board_array.shape != (8, 8):
        raise HTTPException(status_code=400, detail="Board must be 8x8")

    input_tensor = board_array.reshape(1, 8, 8).astype(np.float32)
    inputs = {sess.get_inputs()[0].name: input_tensor}

    outputs = sess.run(None, inputs)
    scores = outputs[0].flatten()

    best_move_index = int(np.argmax(scores))
    x, y = best_move_index % 8, best_move_index // 8

    return {"x": x, "y": y}

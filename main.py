from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# GitHub Pages からのアクセスを許可（CORS設定）
origins = [
    "https://nisi235.github.io",
    "https://nisi235.github.io/othello-ai-apk"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルのパス（Renderの構成に合わせる）
model_paths = {
    "ai1": "models/othello_ai1.onnx",
    "ai2": "models/othello_ai2.onnx",
    "ai3": "models/othello_ai3.onnx",
    "ai4": "models/othello_ai4.onnx",
    "ai5": "models/othello_ai5.onnx",
}

# モデルの読み込み
sessions = {}
for key, path in model_paths.items():
    try:
        sessions[key] = ort.InferenceSession(path)
    except Exception as e:
        print(f"モデル {key} の読み込みに失敗: {e}")

# リクエストのデータ型
class BoardRequest(BaseModel):
    model_key: str  # ai1 ~ ai5
    board: list[list[int]]  # 8x8の盤面

# 動作確認用エンドポイント
@app.get("/")
def root():
    return {"message": "Othello AI API is running"}

# 推論用エンドポイント
@app.post("/predict")
async def predict_next_move(req: BoardRequest):
    if req.model_key not in sessions:
        raise HTTPException(status_code=400, detail="Invalid model_key")

    sess = sessions[req.model_key]
    board_array = np.array(req.board, dtype=np.float32)
    if board_array.shape != (8, 8):
        raise HTTPException(status_code=400, detail="Board must be 8x8")

    input_tensor = board_array.reshape(1, 8, 8)
    inputs = {sess.get_inputs()[0].name: input_tensor}
    outputs = sess.run(None, inputs)
    scores = outputs[0].flatten()

    best_move_index = int(np.argmax(scores))
    x, y = best_move_index % 8, best_move_index // 8

    return {"x": x, "y": y}

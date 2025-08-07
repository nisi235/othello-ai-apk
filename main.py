from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# CORS設定（GitHub PagesのURL）
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

# モデル読み込み
model_paths = {
    "ai1": "models/othello_ai1.onnx",
    "ai2": "models/othello_ai2.onnx",
    "ai3": "models/othello_ai3.onnx",
    "ai4": "models/othello_ai4.onnx",
    "ai5": "models/othello_ai5.onnx",
}

sessions = {}
for key, path in model_paths.items():
    try:
        sessions[key] = ort.InferenceSession(path)
    except Exception as e:
        print(f"モデル読み込み失敗: {key}, {path}, {e}")

# リクエスト形式
class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]

@app.get("/")
def root():
    return {"message": "Othello AI API is running"}

@app.get("/status")
def status():
    return {
        "message": "API is running",
        "loaded_models": [{"key": k, "path": model_paths[k]} for k in sessions.keys()]
    }

# 石を置けるかの判定関数（board[x][y]アクセスに修正）
def can_place(board, x, y, color):
    if board[x][y] != 0:
        return False
    directions = [
        (1,0), (-1,0), (0,1), (0,-1),
        (1,1), (-1,-1), (1,-1), (-1,1)
    ]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found_opponent = False
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board[nx][ny] == -color:
                found_opponent = True
                nx += dx
                ny += dy
            else:
                break
        if found_opponent and 0 <= nx < 8 and 0 <= ny < 8:
            if board[nx][ny] == color:
                return True
    return False

@app.post("/predict")
async def predict_move(req: BoardRequest):
    if req.model_key not in sessions:
        raise HTTPException(status_code=400, detail="Invalid model_key")

    sess = sessions[req.model_key]

    board_array = np.array(req.board, dtype=np.float32)
    if board_array.shape != (8, 8):
        raise HTTPException(status_code=400, detail="Board must be 8x8")

    # 転置して (x,y) に揃える
    board_array_xy = board_array.T

    input_tensor = board_array_xy.reshape(1, 8, 8).astype(np.float32)
    inputs = {sess.get_inputs()[0].name: input_tensor}

    outputs = sess.run(None, inputs)
    scores = outputs[0].flatten()

    sorted_indices = np.argsort(scores)[::-1]
    ai_color = -1  # 白（AI）
    for idx in sorted_indices:
        x, y = idx % 8, idx // 8
        if can_place(board_array_xy, x, y, color=ai_color):
            return {"x": int(x), "y": int(y)}

    return {"x": -1, "y": -1}

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# CORS設定（GitHub PagesのURLに合わせてください）
origins = [
    "https://nisi235.github.io",
    "https://nisi235.github.io/othello-ai-apk",
    "*",  # テスト段階はワイルドカードでも可、本番は限定してください
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class BoardRequest(BaseModel):
    model_key: str  # ai1 ~ ai5
    board: list[list[int]]  # 8x8

@app.get("/")
def root():
    return {"message": "Othello AI API is running"}

@app.get("/status")
def status():
    return {
        "message": "API is running",
        "loaded_models": list(sessions.keys())
    }

def can_place(board, x, y, color):
    if board[y][x] != 0:
        return False
    directions = [
        (1,0), (-1,0), (0,1), (0,-1),
        (1,1), (-1,-1), (1,-1), (-1,1)
    ]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found_opponent = False
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board[ny][nx] == -color:
                found_opponent = True
                nx += dx
                ny += dy
            else:
                break
        if found_opponent and 0 <= nx < 8 and 0 <= ny < 8:
            if board[ny][nx] == color:
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

    input_tensor = board_array.reshape(1, 8, 8).astype(np.float32)
    inputs = {sess.get_inputs()[0].name: input_tensor}

    outputs = sess.run(None, inputs)
    scores = outputs[0].flatten()

    # スコアの高い順に置ける場所を探す
    sorted_indices = np.argsort(scores)[::-1]
    for idx in sorted_indices:
        x, y = idx % 8, idx // 8
        # AIの色は-1（白）
        if can_place(board_array, x, y, color=-1):
            return {"x": x, "y": y}

    # 置ける場所がなければパスとして(-1, -1)を返す
    return {"x": -1, "y": -1}

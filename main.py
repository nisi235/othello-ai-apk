from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# =============================
# CORS設定
# =============================
origins = [
    "https://nisi235.github.io",
    "https://nisi235.github.io/othello-ai-apk"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# モデル読み込み
# =============================
model_paths = {
    "ai1": "models/othello_ai1.onnx",
    "ai2": "models/othello_ai2.onnx",
    "ai3": "models/othello_ai3.onnx",
    "ai4": "models/othello_ai4.onnx",
    "ai5": "models/othello_ai5.onnx",
    "ai6": "models/othello_ai6.onnx",
    "ai7": "models/othello_ai7.onnx",
}

sessions = {}

for key, path in model_paths.items():
    try:
        sessions[key] = ort.InferenceSession(path)
        print(f"モデル読み込み成功: {key}")
    except Exception as e:
        print(f"モデル読み込み失敗: {key}, {path}, {e}")

# =============================
# リクエスト形式
# =============================
class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]

# =============================
# API確認
# =============================
@app.get("/")
def root():
    return {"message": "Othello AI API is running"}

# =============================
# モデル状態確認
# =============================
@app.get("/status")
def status():
    return {
        "message": "API is running",
        "loaded_models": [{"key": k, "path": model_paths[k]} for k in sessions.keys()]
    }

# =============================
# 石を置けるか判定
# =============================
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

# =============================
# AI予測
# =============================
@app.post("/predict")
async def predict_move(req: BoardRequest):

    try:

        print("predict request:", req.model_key)

        if req.model_key not in sessions:
            raise HTTPException(status_code=400, detail="Invalid model_key")

        sess = sessions[req.model_key]

        board_array = np.array(req.board, dtype=np.float32)

        if board_array.shape != (8, 8):
            raise HTTPException(status_code=400, detail="Board must be 8x8")

        board_array_xy = board_array.T

        ai_color = -1

# =============================
# 入力作成
# =============================

flat_board = (board_array_xy * ai_color).flatten()

input_size = sess.get_inputs()[0].shape[1]

# ---- 64入力AI ----
if input_size == 64:

    input_tensor = flat_board.reshape(1,64).astype(np.float32)

# ---- 67入力AI ----
elif input_size == 67:

    piece_diff = np.sum(board_array_xy == ai_color) - np.sum(board_array_xy == -ai_color)
    pos_score = 0
    corner_score = 0

    features = np.array([piece_diff, pos_score, corner_score], dtype=np.float32)

    input_tensor = np.concatenate([flat_board, features]).reshape(1,67).astype(np.float32)

else:
    raise Exception("Unknown model input size")
        inputs = {sess.get_inputs()[0].name: input_tensor}

        outputs = sess.run(None, inputs)

        scores = outputs[0].flatten()
        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:

            y = idx % 8
            x = idx // 8

            if can_place(board_array_xy, x, y, ai_color):
                return {"x": int(x), "y": int(y)}

        return {"x": -1, "y": -1}

    except Exception as e:

        print("predict error:", e)

        return {
            "error": str(e)
        }


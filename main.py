from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# モデルの自動スキャンと構造解析
# =============================
MODEL_DIR = "models"
sessions = {}
model_configs = {} # 各モデルの入力サイズを記録

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

for filename in os.listdir(MODEL_DIR):
    if filename.endswith(".onnx"):
        key = filename.replace(".onnx", "")
        path = os.path.join(MODEL_DIR, filename)
        try:
            sess = ort.InferenceSession(path)
            # 入力サイズを取得 (64か、67か、あるいは(1,1,8,8)か)
            input_shape = sess.get_inputs()[0].shape
            # 形状の最後の次元、または全体の積から入力数を判定
            input_size = np.prod([s for s in input_shape if isinstance(s, int)])
            
            sessions[key] = sess
            model_configs[key] = input_size
            print(f"--- [読込成功] {key} (入力サイズ: {input_size}) ---")
        except Exception as e:
            print(f"--- [読込失敗] {key}, エラー: {e} ---")

class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]

# =============================
# 特徴量抽出（67入力モデル用）
# =============================
def extract_extra_features(board_array, ai_color):
    opp_color = -ai_color
    # 1. 石の数差
    piece_diff = np.sum(board_array == ai_color) - np.sum(board_array == opp_color)
    # 2. 位置重みスコア（簡易版）
    weights = np.array([
        [100,-20,10,5,5,10,-20,100],
        [-20,-50,-2,-2,-2,-2,-50,-20],
        [10,-2,5,1,1,5,-2,10],
        [5,-2,1,0,0,1,-2,5],
        [5,-2,1,0,0,1,-2,5],
        [10,-2,5,1,1,5,-2,10],
        [-20,-50,-2,-2,-2,-2,-50,-20],
        [100,-20,10,5,5,10,-20,100]
    ])
    pos_score = np.sum(weights * (board_array == ai_color)) - np.sum(weights * (board_array == opp_color))
    # 3. 角の数
    corners = [board_array[0,0], board_array[0,7], board_array[7,0], board_array[7,7]]
    corner_score = corners.count(ai_color) - corners.count(opp_color)
    
    return np.array([piece_diff, pos_score, corner_score], dtype=np.float32)

def can_place(board, x, y, color):
    if board[x][y] != 0: return False
    for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        nx, ny = x + dx, y + dy
        found = False
        while 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == -color:
            nx += dx; ny += dy; found = True
        if found and 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == color:
            return True
    return False

@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        key = req.model_key
        if key not in sessions:
            raise HTTPException(status_code=400, detail=f"モデル {key} がありません")

        sess = sessions[key]
        size = model_configs[key]
        board = np.array(req.board, dtype=np.float32)
        ai_color = -1 # 基本的にAIは白(後手)と想定
        
        # モデルの形に合わせて入力を加工
        flat_board = (board * ai_color).flatten()

        if size == 64:
            # シンプルな全結合層モデル (1, 64)
            input_tensor = flat_board.reshape(1, 64)
        elif size == 67:
            # 特徴量付きモデル (1, 67)
            features = extract_extra_features(board, ai_color)
            input_tensor = np.concatenate([flat_board, features]).reshape(1, 67)
        elif size == 1 * 1 * 8 * 8:
            # モンスター版 (CNN) モデル (1, 1, 8, 8)
            input_tensor = (board * ai_color).reshape(1, 1, 8, 8)
        else:
            raise Exception(f"未対応の入力サイズです: {size}")

        # 推論
        outputs = sess.run(None, {sess.get_inputs()[0].name: input_tensor.astype(np.float32)})
        scores = outputs[0].flatten()
        
        # 置ける場所の中からスコア最大を選択
        best_move = {"x": -1, "y": -1}
        indices = np.argsort(scores)[::-1]
        for idx in indices:
            x, y = divmod(idx, 8)
            if can_place(board, x, y, ai_color):
                best_move = {"x": int(x), "y": int(y)}
                break
        return best_move

    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def status():
    return {"loaded_models": model_configs}

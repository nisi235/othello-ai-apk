import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("othello-ai")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"
sessions = {}
model_shapes = {}
model_inputs = {}

# ===============================
# モデル読み込み（ai7.onnx対応）
# ===============================
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

for filename in os.listdir(MODEL_DIR):
    if filename.endswith(".onnx"):
        key = filename.replace(".onnx", "")
        path = os.path.join(MODEL_DIR, filename)
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            sessions[key] = sess
            model_shapes[key] = sess.get_inputs()[0].shape
            model_inputs[key] = sess.get_inputs()[0].name
            logger.info(f"[成功] モデルロード: {key}")
        except Exception as e:
            logger.error(f"[失敗] {filename}: {e}")

class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]
    # AIの色を明示的に受け取る。なければ-1（白）とする。
    ai_color: int = -1

# ===============================
# 超厳格なルールエンジン
# ===============================
DIR = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def get_strict_valid_moves(board, player):
    valid_moves = []
    for r in range(8):
        for c in range(8):
            if board[r][c] != 0: continue
            for dr, dc in DIR:
                nr, nc = r + dr, c + dc
                found_opp = False
                while 0 <= nr < 8 and 0 <= nc < 8:
                    if board[nr][nc] == -player:
                        found_opp = True
                    elif board[nr][nc] == player:
                        if found_opp: valid_moves.append((r, c))
                        break
                    else: break
                    nr += dr; nc += dc
    return list(set(valid_moves))

# ===============================
# 予測エンドポイント
# ===============================
@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        # 1. 盤面とAIの色の確定
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color
        
        # 2. 合法手を先に計算（絶対にここからしか選ばない）
        valid_moves = get_strict_valid_moves(board, ai_color)
        
        if not valid_moves:
            logger.info("AI: 置ける場所がないのでパス")
            return {"x": -1, "y": -1}

        # 3. AI推論（ai7.onnxなどのモデル）
        if req.model_key not in sessions:
            raise HTTPException(status_code=400, detail="Model not found")
        
        sess = sessions[req.model_key]
        shape = model_shapes[req.model_key]
        
        # AIが見る盤面（自分を1にする）
        input_board = (board * ai_color).astype(np.float32)
        
        if len(shape) == 4: # CNN形式 (1,1,8,8)
            input_tensor = input_board.reshape(1, 1, 8, 8)
        else: # Flatten形式 (1,64)
            input_tensor = input_board.flatten().reshape(1, 64)

        outputs = sess.run(None, {model_inputs[req.model_key]: input_tensor})
        scores = np.array(outputs[0]).flatten()

        # 4. 合法手リストの中で、AIの評価が最も高いものを探す
        best_r, best_c = -1, -1
        max_q = -float('inf')

        for r, c in valid_moves:
            idx = r * 8 + c
            score = scores[idx]
            
            # 四隅ボーナス（ロジックの補強）
            if (r, c) in [(0,0), (0,7), (7,0), (7,7)]:
                score += 5.0

            if score > max_q:
                max_q = score
                best_r, best_c = r, c

        # 5. 返却（重要：x=横(col), y=縦(row) にマッピング）
        # フロントエンドが x:横, y:縦 を期待している場合に合わせる
        logger.info(f"AI決定: row={best_r}, col={best_c} (ai_color={ai_color})")
        
        return {
            "x": int(best_c), # 横方向
            "y": int(best_r)  # 縦方向
        }

    except Exception as e:
        logger.error(f"Predict Error: {e}")
        return {"x": -1, "y": -1}

@app.get("/status")
def status():
    return {"models": list(sessions.keys())}

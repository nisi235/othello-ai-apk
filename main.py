import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("othello-monster")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = "models"
sessions = {}
model_shapes = {}
model_inputs = {}

# モデルロード
if os.path.exists(MODEL_DIR):
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".onnx"):
            key = filename.replace(".onnx", "")
            path = os.path.join(MODEL_DIR, filename)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            sessions[key] = sess
            model_shapes[key] = sess.get_inputs()[0].shape
            model_inputs[key] = sess.get_inputs()[0].name
            logger.info(f"Loaded: {key}")

class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]
    ai_color: int = -1 # AIの色（1:黒, -1:白）

# --- 定石座標 ---
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES = [(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)]

# 合法手判定
def get_valid_moves(board, player):
    moves = []
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r in range(8):
        for c in range(8):
            if board[r][c] != 0: continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                found_opp = False
                while 0 <= nr < 8 and 0 <= nc < 8:
                    if board[nr][nc] == -player: found_opp = True
                    elif board[nr][nc] == player:
                        if found_opp: moves.append((r, c))
                        break
                    else: break
                    nr += dr; nc += dc
    return list(set(moves))

@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color
        valid_moves = get_valid_moves(board, ai_color)

        if not valid_moves:
            return {"x": -1, "y": -1}

        # AI推論（ai7.onnx）
        sess = sessions.get(req.model_key)
        if not sess: raise HTTPException(status_code=400, detail="Model Not Found")
        
        # AI視点（自分=1）に変換
        input_data = (board * ai_color).astype(np.float32)
        shape = model_shapes[req.model_key]
        input_tensor = input_data.reshape(shape) # (1,1,8,8) or (1,64)
        
        outputs = sess.run(None, {model_inputs[req.model_key]: input_tensor})
        ai_scores = np.array(outputs[0]).flatten()

        # --- ロジックの詰め込み ---
        best_move = None
        max_total_score = -float('inf')

        for r, c in valid_moves:
            # 1. AIの基本スコア
            score = ai_scores[r * 8 + c]

            # 2. 定石補正（角は最優先、X/Cは減点）
            if (r, c) in CORNERS:
                score += 5000.0  # AIが何と言おうと角は取る
            elif (r, c) in X_SQUARES:
                score -= 3000.0  # 角を奪われるリスク回避
            elif (r, c) in C_SQUARES:
                score -= 1000.0  # 序盤のC打ちは控える

            # 3. 開放度・機動力（相手の選択肢を奪う）
            # その手に打った後の盤面をシミュレート
            temp_board = board.copy()
            # ※簡易的な機動力判定：相手の次手の数を引く
            opp_next_moves = len(get_valid_moves(temp_board, -ai_color))
            score -= opp_next_moves * 0.5 

            # スコア更新
            if score > max_total_score:
                max_total_score = score
                best_move = (r, c)

        # 決定（x=横, y=縦）
        final_r, final_c = best_move
        logger.info(f"AI Selected: Row={final_r}, Col={final_c} | Score={max_total_score}")

        return {
            "x": int(final_c), # 横(Column)
            "y": int(final_r)  # 縦(Row)
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"x": -1, "y": -1}

@app.get("/status")
def status():
    return {"models": list(sessions.keys()), "status": "Monster Logic Online"}

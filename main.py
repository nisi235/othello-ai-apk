import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monster-ultra")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 戦略定数 ---
MODEL_DIR = "models"
sessions = {}
model_shapes = {}
model_inputs = {}

# 盤面評価テーブル (Positional Evaluation)
# 世界中のAIでベースとされる重み。角を100点、その隣を-50点など。
WEIGHT_MATRIX = np.array([
    [ 120, -20,  20,   5,   5,  20, -20, 120],
    [ -20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [  20,  -5,  15,   3,   3,  15,  -5,  20],
    [   5,  -5,   3,   3,   3,   3,  -5,   5],
    [   5,  -5,   3,   3,   3,   3,  -5,   5],
    [  20,  -5,  15,   3,   3,  15,  -5,  20],
    [ -20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 120, -20,  20,   5,   5,  20, -20, 120]
])

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]

# モデルロード
if os.path.exists(MODEL_DIR):
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".onnx"):
            key = filename.replace(".onnx", "")
            sess = ort.InferenceSession(os.path.join(MODEL_DIR, filename))
            sessions[key] = sess
            model_shapes[key] = sess.get_inputs()[0].shape
            model_inputs[key] = sess.get_inputs()[0].name
            logger.info(f"Loaded: {key}")

class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]
    ai_color: int = -1 # 1:黒, -1:白

# --- ヘルパー関数 ---
def get_valid_moves_full(board, player):
    """合法手とその手でひっくり返る石のリストを返す"""
    moves = {}
    for r in range(8):
        for c in range(8):
            if board[r][c] != 0: continue
            flipped = []
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r + dr, c + dc
                line = []
                while 0 <= nr < 8 and 0 <= nc < 8:
                    if board[nr][nc] == -player: line.append((nr, nc))
                    elif board[nr][nc] == player:
                        if line: flipped.extend(line)
                        break
                    else: break
                    nr += dr; nc += dc
            if flipped: moves[(r, c)] = flipped
    return moves

def simulate_move(board, move, flipped, player):
    new_board = board.copy()
    r, c = move
    new_board[r][c] = player
    for fr, fc in flipped:
        new_board[fr][fc] = player
    return new_board

@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color
        valid_moves_dict = get_valid_moves_full(board, ai_color)

        if not valid_moves_dict:
            return {"x": -1, "y": -1}

        # 1. 【超優先】角が取れるなら即決 (AIの計算を待たない)
        for r, c in valid_moves_dict.keys():
            if (r, c) in CORNERS:
                logger.info(f"Corner Take: {r},{c}")
                return {"x": int(c), "y": int(r)}

        # 2. AI推論実行 (ai7.onnx)
        sess = sessions.get(req.model_key)
        if not sess: raise HTTPException(status_code=400, detail="Model Not Found")
        
        input_data = (board * ai_color).astype(np.float32)
        shape = model_shapes[req.model_key]
        outputs = sess.run(None, {model_inputs[req.model_key]: input_data.reshape(shape)})
        ai_scores = np.array(outputs[0]).flatten()

        # 3. 各合法手の評価 (AIスコア + 戦略ロジック)
        best_move = None
        highest_score = -float('inf')
        empty_cells = np.count_nonzero(board == 0)

        for (r, c), flipped in valid_moves_dict.items():
            # (A) AI推論ベーススコア
            current_score = ai_scores[r * 8 + c] * 2.0
            
            # (B) 盤面位置の価値加算
            current_score += WEIGHT_MATRIX[r][c]
            
            # (C) 相手の機動力奪取 (相手の次手の数を減らす)
            future_board = simulate_move(board, (r, c), flipped, ai_color)
            opp_moves_count = len(get_valid_moves_full(future_board, -ai_color))
            current_score -= opp_moves_count * 10.0 # 相手の手を縛る

            # (D) 開放度理論 (周囲に空きマスがある石をひっくり返すのを避ける)
            # 中盤までは石を内側に閉じ込めるほうが強い
            if empty_cells > 15:
                liberty_penalty = 0
                for fr, fc in flipped:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = fr+dr, fc+dc
                        if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == 0:
                            liberty_penalty += 1
                current_score -= liberty_penalty * 8.0

            # (E) 自滅回避 (角が取られていないのにXに打つ手を重罰)
            if (r, c) in X_SQUARES:
                # 対応する角が自分のものor相手のものでない(=空)なら、絶対打たない
                # 角のインデックスを特定
                cr, cc = (0 if r < 4 else 7), (0 if c < 4 else 7)
                if board[cr][cc] == 0:
                    current_score -= 5000.0

            if current_score > highest_score:
                highest_score = current_score
                best_move = (r, c)

        # 決定手 (x=Col, y=Row)
        final_r, final_c = best_move
        logger.info(f"Final Move: Row={final_r}, Col={final_c} | Score={highest_score}")

        return {"x": int(final_c), "y": int(final_r)}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"x": -1, "y": -1}

@app.get("/status")
def status():
    return {"models": list(sessions.keys()), "engine": "Monster Ultra V2"}

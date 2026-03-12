import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monster-ultra-v3")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 戦略定数 ---
MODEL_DIR = "models"
sessions = {}
model_shapes = {}
model_inputs = {}

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
    ai_color: int = -1

# --- ロジック関数 ---

def get_valid_moves_full(board, player):
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

def count_stable_discs(board, player):
    """確定石（二度とひっくり返されない石）を数える"""
    stable = 0
    # 簡易的に4隅から繋がっている石をチェック
    for r, c in CORNERS:
        if board[r][c] == player:
            stable += 1
            # 辺に沿ってどこまで自色が続いているかチェック（簡易版）
            # ここに詳細な探索を入れるとさらに強くなります
    return stable

def evaluate_board_total(board, ai_color, ai_scores_raw):
    """盤面の総合評価（AI推論 + 静的ロジック）"""
    empty_cells = np.count_nonzero(board == 0)
    
    # 1. 重みテーブル評価
    pos_score = np.sum(board * ai_color * WEIGHT_MATRIX)
    
    # 2. 確定石ボーナス
    stable_score = count_stable_discs(board, ai_color) * 50
    
    # 3. 機動力評価（相手の打てる場所が少ないほど良い）
    opp_moves = len(get_valid_moves_full(board, -ai_color))
    mobility_score = -opp_moves * 15
    
    # 4. 偶数理論（最後の手を打てるか）
    parity_score = 20 if empty_cells % 2 == 0 else 0

    return pos_score + stable_score + mobility_score + parity_score

@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color
        valid_moves_dict = get_valid_moves_full(board, ai_color)

        if not valid_moves_dict:
            return {"x": -1, "y": -1}

        # --- 【2手読み Minimax】 ---
        best_move = None
        max_eval = -float('inf')

        # AIの生スコアを取得
        sess = sessions.get(req.model_key)
        input_data = (board * ai_color).astype(np.float32)
        outputs = sess.run(None, {model_inputs[req.model_key]: input_data.reshape(model_shapes[req.model_key])})
        ai_scores = np.array(outputs[0]).flatten()

        for (r, c), flipped in valid_moves_dict.items():
            # 1手目：自分が打つ
            board_after_me = simulate_move(board, (r, c), flipped, ai_color)
            
            # X打ちペナルティ（即時）
            penalty = 0
            if (r, c) in X_SQUARES:
                cr, cc = (0 if r < 4 else 7), (0 if c < 4 else 7)
                if board[cr][cc] == 0: penalty = -5000

            # 2手目：相手の最善の返しを想定
            opp_moves = get_valid_moves_full(board_after_me, -ai_color)
            if opp_moves:
                min_eval_for_me = float('inf')
                for orr, occ in opp_moves.keys():
                    # 相手が角を取れる手があるなら、このルートの評価は最低
                    if (orr, occ) in CORNERS:
                        min_eval_for_me = -10000
                        break
                    # 本来はここで再帰的に計算するが、速度優先で簡易評価
                    eval_val = evaluate_board_total(board_after_me, ai_color, ai_scores)
                    if eval_val < min_eval_for_me:
                        min_eval_for_me = eval_val
                current_total_score = min_eval_for_me + (ai_scores[r*8+c] * 10) + penalty
            else:
                # 相手がパスになる場合は最高！
                current_total_score = 10000 + penalty

            if current_total_score > max_eval:
                max_eval = current_total_score
                best_move = (r, c)

        return {"x": int(best_move[1]), "y": int(best_move[0])}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"x": -1, "y": -1}

@app.get("/status")
def status():
    return {"engine": "Monster Ultra V3 (Minimax + Parity)"}

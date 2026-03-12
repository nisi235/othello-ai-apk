import logging
import os
import numpy as np
import onnxruntime as ort
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monster-ultra-v5")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 戦略定数 ---
MODEL_DIR = "models"
sessions = {}
model_shapes = {}
model_inputs = {}
TT = {}

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

WEIGHT_MATRIX = np.array([
    [ 120, -40,  20,   5,   5,  20, -40, 120],
    [ -40, -80,  -1,  -1,  -1,  -1, -80, -40],
    [  20,  -1,   5,   1,   1,   5,  -1,  20],
    [   5,  -1,   1,   0,   0,   1,  -1,   5],
    [   5,  -1,   1,   0,   0,   1,  -1,   5],
    [  20,  -1,   5,   1,   1,   5,  -1,  20],
    [ -40, -80,  -1,  -1,  -1,  -1, -80, -40],
    [ 120, -40,  20,   5,   5,  20, -40, 120]
])

class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]
    ai_color: int = -1

# --- ヘルパー関数 ---

def get_valid_moves(board, player):
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
    for fr, fc in flipped: new_board[fr][fc] = player
    return new_board

def evaluate_board(board, ai_color):
    """評価関数"""
    pos_score = np.sum(board * ai_color * WEIGHT_MATRIX)
    my_moves = len(get_valid_moves(board, ai_color))
    opp_moves = len(get_valid_moves(board, -ai_color))
    mobility_score = (my_moves - opp_moves) * 20
    return pos_score + mobility_score

# --- PVS 探索 (修正版) ---
def pvs(board, depth, alpha, beta, player, ai_color):
    # 置換表チェック
    board_hash = hash(board.tobytes())
    if board_hash in TT and TT[board_hash]['depth'] >= depth:
        return TT[board_hash]['score'], TT[board_hash]['move']

    moves = get_valid_moves(board, player)
    
    # パスまたは終局の処理
    if not moves:
        opp_moves = get_valid_moves(board, -player)
        if not opp_moves: # 終局
            return np.sum(board * ai_color) * 100, None
        # パス
        score, _ = pvs(board, depth - 1, -beta, -alpha, -player, ai_color)
        return -score, None

    if depth == 0:
        return evaluate_board(board, ai_color), None

    best_move = None
    for i, (move, flipped) in enumerate(moves.items()):
        new_board = simulate_move(board, move, flipped, player)
        
        if i == 0: # 最初の枝はフルウィンドウ探索
            score, _ = pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color)
            score = -score
        else: # それ以降はNull Window探索
            score, _ = pvs(new_board, depth - 1, -alpha - 1, -alpha, -player, ai_color)
            score = -score
            if alpha < score < beta:
                score, _ = pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color)
                score = -score

        if score > alpha:
            alpha = score
            best_move = move
        if alpha >= beta: break 

    TT[board_hash] = {'score': alpha, 'depth': depth, 'move': best_move}
    return alpha, best_move

@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color
        empty_cells = np.count_nonzero(board == 0)
        
        if len(TT) > 10000: TT.clear()

        # 1. 終盤読み切り（負荷軽減のため12手に設定）
        if empty_cells <= 12:
            _, move = pvs(board, empty_cells, -100000, 100000, ai_color, ai_color)
            if move: return {"x": int(move[1]), "y": int(move[0])}

        # 2. 中盤探索
        valid_moves = get_valid_moves(board, ai_color)
        if not valid_moves: return {"x": -1, "y": -1}

        # 4手読み実行
        _, move = pvs(board, 4, -100000, 100000, ai_color, ai_color)
        
        if move:
            return {"x": int(move[1]), "y": int(move[0])}
        else:
            # 万が一Noneが返った場合のフォールバック
            first_move = list(valid_moves.keys())[0]
            return {"x": int(first_move[1]), "y": int(first_move[0])}

    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return {"x": -1, "y": -1}

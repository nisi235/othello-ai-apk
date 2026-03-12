import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monster-ultra-v6")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- グローバル設定 ---
MODEL_DIR = "models"
sessions, model_shapes, model_inputs = {}, {}, {}
TT = {} # 置換表

# モデルロード
if os.path.exists(MODEL_DIR):
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".onnx"):
            key = filename.replace(".onnx", "")
            sess = ort.InferenceSession(os.path.join(MODEL_DIR, filename))
            sessions[key] = sess
            model_shapes[key] = sess.get_inputs()[0].shape
            model_inputs[key] = sess.get_inputs()[0].name

# 戦略重み表（中盤の指針）
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

# --- ロジックコア ---

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

def get_frontier_score(board, flipped):
    """開放度理論：ひっくり返した石の周囲にどれだけ空きマスがあるか"""
    frontier_count = 0
    for fr, fc in flipped:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == 0:
                frontier_count += 1
    return frontier_count

def evaluate_board(board, ai_color):
    """パス強制に特化した評価関数"""
    # 1. 位置評価
    pos_score = np.sum(board * ai_color * WEIGHT_MATRIX)
    
    # 2. 強烈な機動力評価（Mobility）
    my_moves = get_valid_moves(board, ai_color)
    opp_moves = get_valid_moves(board, -ai_color)
    
    m_count = len(my_moves)
    o_count = len(opp_moves)
    
    # 相手をパスに追い込んだら超巨大ボーナス
    mobility_score = m_count * 20
    if o_count == 0:
        mobility_score += 1500 
    else:
        mobility_score -= o_count * 50 # 相手の選択肢を奪う重みを最大化
        
    return pos_score + mobility_score

def pvs(board, depth, alpha, beta, player, ai_color):
    board_hash = hash(board.tobytes())
    if board_hash in TT and TT[board_hash]['depth'] >= depth:
        return TT[board_hash]['score'], TT[board_hash]['move']

    moves = get_valid_moves(board, player)
    if not moves:
        opp_moves = get_valid_moves(board, -player)
        if not opp_moves: return np.sum(board * ai_color) * 100, None
        score, _ = pvs(board, depth - 1, -beta, -alpha, -player, ai_color)
        return -score, None

    if depth == 0:
        return evaluate_board(board, ai_color), None

    best_move = None
    # 探索順の最適化：角を優先、X打ちを後回し
    sorted_moves = sorted(moves.items(), key=lambda x: WEIGHT_MATRIX[x[0][0]][x[0][1]], reverse=True)

    for i, (move, flipped) in enumerate(sorted_moves):
        new_board = simulate_move(board, move, flipped, player)
        # 解放度ペナルティ（中盤のみ：自分の石を露出させない）
        f_penalty = get_frontier_score(board, flipped) * 5 if depth > 2 else 0
        
        if i == 0:
            score, _ = pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color)
            score = -score - f_penalty
        else:
            score, _ = pvs(new_board, depth - 1, -alpha - 1, -alpha, -player, ai_color)
            score = -score - f_penalty
            if alpha < score < beta:
                score, _ = pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color)
                score = -score - f_penalty

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

        # 終盤12手：完璧に仕留める
        if empty_cells <= 12:
            _, move = pvs(board, empty_cells, -100000, 100000, ai_color, ai_color)
            if move: return {"x": int(move[1]), "y": int(move[0])}

        # 中盤：相手の自由を奪う（3手読みで速度維持）
        _, move = pvs(board, 3, -100000, 100000, ai_color, ai_color)
        
        if not move:
            valid = get_valid_moves(board, ai_color)
            move = list(valid.keys())[0] if valid else (-1, -1)

        return {"x": int(move[1]), "y": int(move[0])}

    except Exception as e:
        return {"x": -1, "y": -1}

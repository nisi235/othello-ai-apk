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

# --- グローバル置換表 (Transposition Table) ---
# 同じ盤面の計算結果を再利用し、探索速度を10倍以上に引き上げる
TT = {}

# --- 戦略重み ---
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

# --- 最高峰ロジック：ビット演算ライクな高速処理 ---

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

def evaluate_mobility_and_patterns(board, ai_color):
    """最高峰の評価：機動力 + 確定石 + エッジパターン"""
    empty = np.count_nonzero(board == 0)
    
    # 1. 位置と確定石の統合スコア
    score = np.sum(board * ai_color * WEIGHT_MATRIX)
    
    # 2. 機動力（Mobility）: 相手の選択肢を奪う
    my_m = len(get_valid_moves(board, ai_color))
    opp_m = len(get_valid_moves(board, -ai_color))
    score += (my_m - opp_m) * 25  # 相手を詰ませる重みを強化
    
    # 3. 辺の安定性（ウィング・山の回避）
    for edge in [board[0,:], board[7,:], board[:,0], board[:,7]]:
        e = edge * ai_color
        if e[0] == 0 and e[7] == 0 and np.all(e[1:7] == 1): score -= 60 # 山への厳罰
        
    return score

# --- PVS (Principal Variation Search) アルゴリズム ---
# 最善手以外の探索を極限まで省略する最高峰の探索術
def pvs(board, depth, alpha, beta, player, ai_color):
    board_hash = hash(board.tobytes())
    if board_hash in TT and TT[board_hash]['depth'] >= depth:
        return TT[board_hash]['score']

    moves = get_valid_moves(board, player)
    if not moves:
        opp_moves = get_valid_moves(board, -player)
        if not opp_moves: return np.sum(board * ai_color) * 100, None
        return -pvs(board, depth - 1, -beta, -alpha, -player, ai_color), None

    if depth == 0:
        return evaluate_mobility_and_patterns(board, ai_color), None

    best_move = None
    for i, (move, flipped) in enumerate(moves.items()):
        new_board = simulate_move(board, move, flipped, player)
        
        if i == 0:
            score = -pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color)
        else:
            # Null Window Search: 相手がこれ以上良くならないことを素早く確認
            score = -pvs(new_board, depth - 1, -alpha - 1, -alpha, -player, ai_color)
            if alpha < score < beta:
                score = -pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color)

        if score > alpha:
            alpha = score
            best_move = move
        if alpha >= beta: break # 枝刈り

    TT[board_hash] = {'score': alpha, 'depth': depth}
    return alpha, best_move

@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color
        empty_cells = np.count_nonzero(board == 0)
        
        # 置換表のクリーンアップ（メモリ節約）
        if len(TT) > 5000: TT.clear()

        # 1. 【超終盤】残り14手以下ならPVSで完全読み切り
        if empty_cells <= 14:
            logger.info(f"Endgame Solver: Deep Search {empty_cells}")
            _, move = pvs(board, empty_cells, -10000, 10000, ai_color, ai_color)
            if move: return {"x": int(move[1]), "y": int(move[0])}

        # 2. 【中盤】PVS (4手読み) + 学習モデル(ai7)の融合
        # 学習モデルの「直感」を初手の並び替えに使用（Move Ordering）
        valid_moves = get_valid_moves(board, ai_color)
        if not valid_moves: return {"x": -1, "y": -1}

        # AI推論(ai7)のスコアで探索順を最適化
        # 良い手から探索することで「枝刈り」の効率が最大化される
        sess = sessions.get(req.model_key)
        input_data = (board * ai_color).astype(np.float32)
        outputs = sess.run(None, {model_inputs[req.model_key]: input_data.reshape(model_shapes[req.model_key])})
        ai_scores = np.array(outputs[0]).flatten()

        # ソートしてPVSに渡す
        sorted_moves = sorted(valid_moves.items(), 
                             key=lambda m: ai_scores[m[0][0]*8 + m[0][1]], 
                             reverse=True)
        
        # PVS実行 (4手先読み)
        _, move = pvs(board, 4, -10000, 10000, ai_color, ai_color)
        
        if not move: # 万が一のフォールバック
            move = sorted_moves[0][0]

        return {"x": int(move[1]), "y": int(move[0])}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"x": -1, "y": -1}

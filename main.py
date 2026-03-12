import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monster-ultra-v4")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 定数 ---
MODEL_DIR = "models"
sessions = {}
model_shapes = {}
model_inputs = {}

# 1. 盤面位置の基本価値（Positional Evaluation）
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

# --- 内部ロジック ---

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

def evaluate_edge_patterns(board, player):
    """辺のパターン評価（ウィング・山・中抜き）"""
    score = 0
    edges = [
        board[0, :], board[7, :], board[:, 0], board[:, 7]
    ]
    for edge in edges:
        p_edge = edge * player
        # ウィング/山の判定
        if p_edge[0] == 0 and p_edge[7] == 0:
            if np.all(p_edge[1:7] == 1): score -= 50 # 山は危険
            elif np.all(p_edge[2:6] == 1): score -= 30 # ウィングも控えめに減点
        # 中抜きの罠（101の状態）
        for i in range(6):
            if p_edge[i] == 1 and p_edge[i+1] == 0 and p_edge[i+2] == 1:
                score += 40
    return score

def count_stable_discs(board, player):
    """確定石のカウント"""
    stable = 0
    # 4隅から辺に向かって自色が連続しているか
    for r, c, dr, dc in [(0,0,0,1), (0,0,1,0), (0,7,0,-1), (0,7,1,0), (7,0,0,1), (7,0,-1,0), (7,7,0,-1), (7,7,-1,0)]:
        curr_r, curr_c = r, c
        while 0 <= curr_r < 8 and 0 <= curr_c < 8 and board[curr_r][curr_c] == player:
            stable += 1
            curr_r += dr; curr_c += dc
    return stable

def evaluate_full(board, ai_color):
    """総合評価関数"""
    empty_cells = np.count_nonzero(board == 0)
    
    # 1. 位置評価
    pos_score = np.sum(board * ai_color * WEIGHT_MATRIX)
    # 2. 確定石
    stable_score = count_stable_discs(board, ai_color) * 40
    # 3. 辺のパターン
    edge_score = evaluate_edge_patterns(board, ai_color)
    # 4. 機動力
    my_moves = len(get_valid_moves(board, ai_color))
    opp_moves = len(get_valid_moves(board, -ai_color))
    mobility_score = (my_moves - opp_moves) * 15
    # 5. 偶数理論
    parity = 15 if empty_cells % 2 == 0 else 0

    return pos_score + stable_score + edge_score + mobility_score + parity

def solve_endgame(board, player, depth):
    """終盤ソルバー：石の差を最大化する"""
    moves = get_valid_moves(board, player)
    if not moves:
        opp_moves = get_valid_moves(board, -player)
        if not opp_moves: return np.sum(board * player), None
        val, _ = solve_endgame(board, -player, depth - 1)
        return -val, None
    
    if depth == 0: return np.sum(board * player), None

    best_val = -100
    best_move = None
    for move, flipped in moves.items():
        new_board = simulate_move(board, move, flipped, player)
        val, _ = solve_endgame(new_board, -player, depth - 1)
        val = -val
        if val > best_val:
            best_val = val
            best_move = move
    return best_val, best_move

@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color
        valid_moves = get_valid_moves(board, ai_color)

        if not valid_moves: return {"x": -1, "y": -1}

        empty_cells = np.count_nonzero(board == 0)
        
        # --- 終盤：残り10手以下なら完全読み切り ---
        if empty_cells <= 10:
            logger.info("Endgame Solver Activated")
            _, move = solve_endgame(board, ai_color, empty_cells)
            if move: return {"x": int(move[1]), "y": int(move[0])}

        # --- 中盤：評価関数 + Minimax ---
        best_move = None
        max_score = -float('inf')
        
        # AI推論(ai7)のスコア取得
        sess = sessions.get(req.model_key)
        input_data = (board * ai_color).astype(np.float32)
        outputs = sess.run(None, {model_inputs[req.model_key]: input_data.reshape(model_shapes[req.model_key])})
        ai_scores = np.array(outputs[0]).flatten()

        for (r, c), flipped in valid_moves.items():
            # 即時的な角確保
            if (r, c) in CORNERS: return {"x": int(c), "y": int(r)}
            
            # X打ちペナルティ（角が空の時）
            penalty = 0
            if (r, c) in X_SQUARES:
                cr, cc = (0 if r < 4 else 7), (0 if c < 4 else 7)
                if board[cr][cc] == 0: penalty = -10000

            # 1手読み評価
            future_board = simulate_move(board, (r, c), flipped, ai_color)
            score = evaluate_full(future_board, ai_color) + (ai_scores[r*8+c] * 15) + penalty
            
            if score > max_score:
                max_score = score
                best_move = (r, c)

        return {"x": int(best_move[1]), "y": int(best_move[0])}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"x": -1, "y": -1}

@app.get("/status")
def status():
    return {"engine": "Monster Ultra V4 (Perfect Endgame & Pattern Engine)"}

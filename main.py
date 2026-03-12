import logging
import os
import numpy as np
import onnxruntime as ort
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monster-ultra-v8")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- グローバル設定 ---
MODEL_DIR = "models"
sessions, model_shapes, model_inputs = {}, {}, {}
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

# 戦略重み表（角・辺・X/C打ちの基本評価）
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

# --- コアロジック ---

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
    """
    【パス強制・制圧型評価関数】
    石の数ではなく、相手の絶望（手数の少なさ）をスコアにする
    """
    # 1. 基本位置
    pos_score = np.sum(board * ai_color * WEIGHT_MATRIX)
    
    # 2. 強烈な機動力評価（Mobility）
    my_moves = get_valid_moves(board, ai_color)
    opp_moves = get_valid_moves(board, -ai_color)
    
    m_count = len(my_moves)
    o_count = len(opp_moves)
    
    # 相手をパス(0)に追い込めば勝利確定級の加点
    # 自分の手数が多いほど、相手の手数が少ないほど良い
    mobility_score = (m_count * 25) - (o_count * 60)
    if o_count == 0:
        mobility_score += 2000 # パス強制ボーナス（最大級）
        
    # 3. 開放度ペナルティ（自分の石を外に晒さない）
    # 相手に打てる場所を与えないための「内側の守り」
    return pos_score + mobility_score

def pvs(board, depth, alpha, beta, player, ai_color, start_time, time_limit):
    # タイムアウト検知
    if time.time() - start_time > time_limit:
        raise TimeoutError

    board_hash = hash(board.tobytes())
    if board_hash in TT and TT[board_hash]['depth'] >= depth:
        return TT[board_hash]['score'], TT[board_hash]['move']

    moves = get_valid_moves(board, player)
    if not moves:
        opp_moves = get_valid_moves(board, -player)
        if not opp_moves: return np.sum(board * ai_color) * 100, None
        score, _ = pvs(board, depth - 1, -beta, -alpha, -player, ai_color, start_time, time_limit)
        return -score, None

    if depth == 0:
        return evaluate_board(board, ai_color), None

    best_move = None
    # 探索効率を上げるための並び替え（Move Ordering）
    sorted_moves = sorted(moves.items(), key=lambda x: WEIGHT_MATRIX[x[0][0]][x[0][1]], reverse=True)

    for i, (move, flipped) in enumerate(sorted_moves):
        new_board = simulate_move(board, move, flipped, player)
        
        if i == 0:
            score, _ = pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color, start_time, time_limit)
            score = -score
        else:
            score, _ = pvs(new_board, depth - 1, -alpha - 1, -alpha, -player, ai_color, start_time, time_limit)
            score = -score
            if alpha < score < beta:
                score, _ = pvs(new_board, depth - 1, -beta, -alpha, -player, ai_color, start_time, time_limit)
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
        
        start_time = time.time()
        # Renderの無料枠制限を考慮し、2.0秒をリミットに設定
        time_limit = 2.0 
        
        if len(TT) > 10000: TT.clear()

        last_best_move = None
        valid_moves = get_valid_moves(board, ai_color)
        if not valid_moves: return {"x": -1, "y": -1}
        
        # 初期値（最悪でも1手は返す）
        last_best_move = list(valid_moves.keys())[0]

        # --- 反復深化：パス強制の精度を時間一杯まで高める ---
        try:
            # 終盤10手なら全読み、中盤は時間切れまで深く
            max_depth = empty_cells if empty_cells <= 10 else 10 
            for depth in range(1, max_depth + 1):
                _, move = pvs(board, depth, -100000, 100000, ai_color, ai_color, start_time, time_limit)
                if move: last_best_move = move
                
                # すでに全読み（終盤）が終わったならループ終了
                if empty_cells <= 10 and depth == empty_cells:
                    break
        except TimeoutError:
            logger.info(f"Time limit reached. Best move at depth {depth-1} returned.")

        return {"x": int(last_best_move[1]), "y": int(last_best_move[0])}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"x": -1, "y": -1}

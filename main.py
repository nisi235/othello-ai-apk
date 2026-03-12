import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ログ設定
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
# モデル読み込み
# ===============================
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logger.info("=== モデル読み込み開始 ===")
for filename in os.listdir(MODEL_DIR):
    if filename.endswith(".onnx"):
        key = filename.replace(".onnx", "")
        path = os.path.join(MODEL_DIR, filename)
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            input_info = sess.get_inputs()[0]
            sessions[key] = sess
            model_shapes[key] = input_info.shape
            model_inputs[key] = input_info.name
            logger.info(f"[成功] {key} shape={input_info.shape}")
        except Exception as e:
            logger.error(f"[失敗] {filename} {e}")
logger.info("=== モデル読み込み完了 ===")

# ===============================
# データ定義
# ===============================
class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]
    # AIの色（1:黒, -1:白）。リクエストにない場合は盤面の石の数から推論するロジックを入れます。
    ai_color: int = -1 

DIR = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ===============================
# ルールエンジン（合法手判定）
# ===============================
def get_valid_moves(board, player):
    moves = []
    for x in range(8):
        for y in range(8):
            if board[x, y] != 0: continue
            for dx, dy in DIR:
                nx, ny = x + dx, y + dy
                found_enemy = False
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if board[nx, ny] == -player:
                        found_enemy = True
                    elif board[nx, ny] == player:
                        if found_enemy: moves.append((x, y))
                        break
                    else: break
                    nx += dx
                    ny += dy
    return list(set(moves))

# ===============================
# AI予測（メインロジック）
# ===============================
@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        logger.info(f"--- predict呼び出し model={req.model_key} ---")
        board = np.array(req.board, dtype=np.int8)
        ai_color = req.ai_color # フロントから送られてきたAIの色

        # 1. まず「ルール上の合法手」をリストアップ
        valid_list = get_valid_moves(board, ai_color)
        
        if not valid_list:
            logger.warning("合法手なし（パス）")
            return {"x": -1, "y": -1}

        # 2. モデル入力の準備
        if req.model_key not in sessions:
            raise HTTPException(status_code=400, detail="モデル未ロード")

        sess = sessions[req.model_key]
        shape = model_shapes[req.model_key]
        
        # AI視点に正規化（自分が1、相手が-1）
        norm_board = board * ai_color
        
        # モデルの形に合わせる (1, 1, 8, 8)
        if len(shape) == 4:
            input_tensor = norm_board.reshape(1, 1, 8, 8).astype(np.float32)
        else:
            input_tensor = norm_board.flatten().reshape(1, 64).astype(np.float32)

        # 3. AI推論実行
        outputs = sess.run(None, {model_inputs[req.model_key]: input_tensor})
        scores = np.array(outputs[0]).flatten() # 64マスのスコア

        # 4. 【最重要】合法手の中からスコアが最大のものを選ぶ
        best_move = None
        max_q = -float('inf')

        # デバッグ用に上位候補をログ出し
        order = np.argsort(scores)[::-1]
        logger.info(f"AI上位候補スコア: {order[:5]}")

        for move in valid_list:
            x, y = move
            idx = x * 8 + y
            score = scores[idx]

            # 角へのボーナス（必要に応じて）
            if (x, y) in [(0,0), (0,7), (7,0), (7,7)]:
                score += 10.0 # 学習が足りない場合の補助

            if score > max_q:
                max_q = score
                best_move = move

        if best_move:
            logger.info(f"決定手: {best_move} (score: {max_q})")
            return {"x": int(best_move[0]), "y": int(best_move[1])}
        
        # 万が一選べなかった場合のフォールバック
        return {"x": int(valid_list[0][0]), "y": int(valid_list[0][1])}

    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {"loaded_models": list(sessions.keys()), "model_count": len(sessions)}

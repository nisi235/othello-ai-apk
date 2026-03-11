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

            sess = ort.InferenceSession(
                path,
                providers=["CPUExecutionProvider"]
            )

            input_info = sess.get_inputs()[0]

            sessions[key] = sess
            model_shapes[key] = input_info.shape
            model_inputs[key] = input_info.name

            logger.info(f"[成功] {key} shape={input_info.shape}")

        except Exception as e:

            logger.error(f"[失敗] {filename} {e}")

logger.info("=== モデル読み込み完了 ===")

# ===============================
# API入力
# ===============================

class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]

DIR = [
(-1,-1),(-1,0),(-1,1),
(0,-1),(0,1),
(1,-1),(1,0),(1,1)
]

# ===============================
# 合法手判定
# ===============================

def can_place(board, x, y, color):

    if board[x][y] != 0:
        return False

    for dx, dy in DIR:

        nx = x + dx
        ny = y + dy
        found_enemy = False

        while 0 <= nx < 8 and 0 <= ny < 8:

            if board[nx][ny] == -color:
                found_enemy = True

            elif board[nx][ny] == color:
                if found_enemy:
                    return True
                break

            else:
                break

            nx += dx
            ny += dy

    return False


# ===============================
# 追加特徴量
# ===============================

def extract_extra_features(board,ai):

    opp = -ai

    piece_diff = np.sum(board==ai) - np.sum(board==opp)

    weights = np.array([
    [100,-20,10,5,5,10,-20,100],
    [-20,-50,-2,-2,-2,-2,-50,-20],
    [10,-2,5,1,1,5,-2,10],
    [5,-2,1,0,0,1,-2,5],
    [5,-2,1,0,0,1,-2,5],
    [10,-2,5,1,1,5,-2,10],
    [-20,-50,-2,-2,-2,-2,-50,-20],
    [100,-20,10,5,5,10,-20,100]
    ])

    pos_score = np.sum(weights*(board==ai)) - np.sum(weights*(board==opp))

    corners = [board[0,0],board[0,7],board[7,0],board[7,7]]

    corner_score = corners.count(ai) - corners.count(opp)

    return np.array([piece_diff,pos_score,corner_score],dtype=np.float32)


# ===============================
# AI予測
# ===============================

@app.post("/predict")
async def predict_move(req:BoardRequest):

    try:

        logger.info(f"predict呼び出し model={req.model_key}")

        board = np.array(req.board,dtype=np.int8)

        logger.info("\n盤面\n" + str(board))

        if req.model_key not in sessions:

            logger.error("モデルが存在しません")
            raise HTTPException(status_code=400,detail="モデルがありません")

        if board.shape != (8,8):

            logger.error(f"盤面サイズエラー {board.shape}")
            raise HTTPException(status_code=400,detail="盤面は8x8必要")

        # 盤面値チェック
        if not np.isin(board,[-1,0,1]).all():

            raise HTTPException(
                status_code=400,
                detail="盤面値は -1 0 1 のみ"
            )

        board = board.copy()

        sess = sessions[req.model_key]
        shape = model_shapes[req.model_key]

        ai_color = -1

        flat = (board * ai_color).flatten()

        # =========================
        # 入力自動判定
        # =========================

        if len(shape)==2 and shape[1]==64:

            input_tensor = flat.reshape(1,64)

        elif len(shape)==2 and shape[1]==67:

            features = extract_extra_features(board,ai_color)
            input_tensor = np.concatenate([flat,features]).reshape(1,67)

        elif len(shape)==4:

            input_tensor = (board * ai_color).reshape(1,1,8,8)

        else:

            raise Exception(f"未対応入力shape {shape}")

        outputs = sess.run(
            None,
            {model_inputs[req.model_key]:input_tensor.astype(np.float32)}
        )

        scores = np.array(outputs[0]).flatten()

        scores = np.nan_to_num(scores,-9999)

        if len(scores) != 64:

            logger.error(f"AI出力サイズ異常 {len(scores)}")
            return {"x":-1,"y":-1}

        # =========================
        # 角優先
        # =========================

        corners = [(0,0),(0,7),(7,0),(7,7)]

        for x,y in corners:

            if can_place(board,x,y,ai_color):

                logger.info(f"角取得 {x},{y}")

                return {"x":x,"y":y}

        # =========================
        # AI候補
        # =========================

        order = np.argsort(scores)[::-1]

        logger.info(f"AI上位候補 index {order[:5]}")

        X_SQUARES = [(1,1),(1,6),(6,1),(6,6)]

        for idx in order:

            if idx < 0 or idx >= 64:
                continue

            x = idx // 8
            y = idx % 8

            if (x,y) in X_SQUARES:
                continue

            if can_place(board,x,y,ai_color):

                logger.info(f"AI選択手 {x},{y}")

                return {"x":int(x),"y":int(y)}

        # =========================
        # fallback
        # =========================

        logger.warning("AI候補失敗 fallback開始")

        for x in range(8):
            for y in range(8):

                if can_place(board,x,y,ai_color):

                    logger.info(f"fallback手 {x},{y}")

                    return {"x":x,"y":y}

        logger.warning("合法手なし")

        return {"x":-1,"y":-1}

    except Exception as e:

        logger.error(f"predictエラー {e}")

        raise HTTPException(status_code=500,detail=str(e))


# ===============================
# 状態確認
# ===============================

@app.get("/status")
def status():

    logger.info("status確認")

    return {

        "loaded_models":model_shapes,
        "model_count":len(model_shapes)

    }


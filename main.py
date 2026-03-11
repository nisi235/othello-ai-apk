import logging
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===============================
# ログ設定（Renderログに表示）
# ===============================

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

            logger.info(f"読み込み中: {filename}")

            sess = ort.InferenceSession(
                path,
                providers=["CPUExecutionProvider"]
            )

            shape = sess.get_inputs()[0].shape

            sessions[key] = sess
            model_shapes[key] = shape

            logger.info(f"[成功] {key} shape={shape}")

        except Exception as e:

            logger.error(f"[失敗] {filename} エラー={e}")

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
# オセロ合法手判定
# ===============================

def can_place(board,x,y,color):

    if board[x][y] != 0:
        return False

    for dx,dy in DIR:

        nx = x + dx
        ny = y + dy

        found = False

        while 0<=nx<8 and 0<=ny<8 and board[nx][ny] == -color:

            nx += dx
            ny += dy
            found = True

        if found and 0<=nx<8 and 0<=ny<8 and board[nx][ny] == color:

            return True

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

        if req.model_key not in sessions:

            logger.error("モデルが存在しません")
            raise HTTPException(status_code=400,detail="モデルがありません")

        board = np.array(req.board,dtype=np.float32)

        if board.shape != (8,8):

            logger.error(f"盤面サイズエラー {board.shape}")
            raise HTTPException(status_code=400,detail="盤面は8x8必要")

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

            logger.error(f"未対応shape {shape}")
            raise Exception(f"未対応入力shape {shape}")

        outputs = sess.run(
            None,
            {sess.get_inputs()[0].name:input_tensor.astype(np.float32)}
        )

        scores = outputs[0].flatten()

        order = np.argsort(scores)[::-1]

        for idx in order:

            x = idx // 8
            y = idx % 8

            if can_place(board,x,y,ai_color):

                logger.info(f"AI選択手 {x},{y}")

                return {"x":int(x),"y":int(y)}

        logger.warning("合法手なし")

        return {"x":-1,"y":-1}

    except Exception as e:

        logger.error(f"predictエラー {e}")

        return {"error":str(e)}


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

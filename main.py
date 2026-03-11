from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import os

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

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("=== モデル読み込み開始 ===")

for filename in os.listdir(MODEL_DIR):

    if filename.endswith(".onnx"):

        key = filename.replace(".onnx","")
        path = os.path.join(MODEL_DIR,filename)

        try:

            sess = ort.InferenceSession(path)

            shape = sess.get_inputs()[0].shape

            sessions[key] = sess
            model_shapes[key] = shape

            print(f"[成功] {key} 入力shape={shape}")

        except Exception as e:

            print(f"[失敗] {filename} エラー {e}")

print("=== モデル読み込み完了 ===")


class BoardRequest(BaseModel):

    model_key : str
    board : list[list[int]]


DIR = [
(-1,-1),(-1,0),(-1,1),
(0,-1),(0,1),
(1,-1),(1,0),(1,1)
]


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


@app.post("/predict")

async def predict_move(req:BoardRequest):

    try:

        key = req.model_key

        if key not in sessions:
            raise HTTPException(status_code=400,detail="モデルがありません")

        sess = sessions[key]

        shape = model_shapes[key]

        board = np.array(req.board,dtype=np.float32)

        ai_color = -1

        flat = (board * ai_color).flatten()

        # =========================
        # 入力形式自動判定
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

        outputs = sess.run(None,{sess.get_inputs()[0].name:input_tensor.astype(np.float32)})

        scores = outputs[0].flatten()

        best_move = {"x":-1,"y":-1}

        order = np.argsort(scores)[::-1]

        for idx in order:

            x = idx // 8
            y = idx % 8

            if can_place(board,x,y,ai_color):

                best_move = {"x":int(x),"y":int(y)}
                break

        return best_move

    except Exception as e:

        return {"error":str(e)}


@app.get("/status")

def status():

    return {

        "loaded_models":model_shapes

    }

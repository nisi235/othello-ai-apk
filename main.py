from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import os

app = FastAPI()

# =============================
# CORS設定（どこからでも接続可能に）
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# モデル読み込み（フォルダ内の.onnxファイルを自動認識）
# =============================
# 'models' フォルダの中に昨日変換した .onnx を入れてください
# ファイル名を othello_ai_monster.onnx にすると "monster" で呼び出せます
MODEL_DIR = "models"
sessions = {}

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 起動時にmodelsフォルダ内のファイルをスキャン
for filename in os.listdir(MODEL_DIR):
    if filename.endswith(".onnx"):
        key = filename.replace(".onnx", "")
        path = os.path.join(MODEL_DIR, filename)
        try:
            sessions[key] = ort.InferenceSession(path)
            print(f"--- [成功] モデル読み込み: {key} ---")
        except Exception as e:
            print(f"--- [失敗] モデル読み込み: {key}, エラー: {e} ---")

# =============================
# リクエスト形式
# =============================
class BoardRequest(BaseModel):
    model_key: str
    board: list[list[int]]

# =============================
# 石を置けるか判定
# =============================
def can_place(board, x, y, color):
    if board[x][y] != 0: return False
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found = False
        while 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == -color:
            nx += dx; ny += dy; found = True
        if found and 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == color:
            return True
    return False

# =============================
# API確認用エンドポイント
# =============================
@app.get("/")
def root():
    return {"message": "モンスターAI API 稼働中", "models": list(sessions.keys())}

# =============================
# AI予測メイン処理
# =============================
@app.post("/predict")
async def predict_move(req: BoardRequest):
    try:
        if req.model_key not in sessions:
            raise HTTPException(status_code=400, detail=f"モデル '{req.model_key}' が見つかりません")

        sess = sessions[req.model_key]
        board_array = np.array(req.board, dtype=np.float32) # (8, 8)
        
        # モンスター版AI（CNN）は自分の石を「1」として考えるため、
        # AIが白(-1)なら、盤面全体に-1を掛けて「自分が1」の状態にする
        ai_color = -1 
        input_board = (board_array * ai_color).reshape(1, 1, 8, 8).astype(np.float32)

        # 推論実行
        inputs = {sess.get_inputs()[0].name: input_board}
        outputs = sess.run(None, inputs)
        scores = outputs[0].flatten() # 64マスの評価値

        # 評価の高い順に「置ける場所」を探す
        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices:
            x, y = divmod(idx, 8)
            if can_place(board_array, x, y, ai_color):
                return {"x": int(x), "y": int(y)}

        return {"x": -1, "y": -1} # 置ける場所がない場合

    except Exception as e:
        print(f"予測エラー: {e}")
        return {"error": str(e)}

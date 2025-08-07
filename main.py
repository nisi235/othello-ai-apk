from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS許可（フロントが別ドメインの場合必須）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # セキュリティ考慮して必要に応じて制限してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AIモデルファイルのパス（modelsフォルダに配置）
model_paths = {
    "ai1": "models/othello_ai1.onnx",
    "ai2": "models/othello_ai2.onnx",
    "ai3": "models/othello_ai3.onnx",
    "ai4": "models/othello_ai4.onnx",
    "ai5": "models/othello_ai5.onnx",
}

# ONNXセッションをロードしてキャッシュ
sessions = {}
for key, path in model_paths.items():
    try:
        sessions[key] = ort.InferenceSession(path)
        print(f"Loaded model {key} from {path}")
    except Exception as e:
        print(f"Failed to load model {key} from {path}: {e}")

# リクエストボディの定義
class BoardRequest(BaseModel):
    model_key: str  # "ai1"～"ai5"
    board: list[list[int]]  # 8x8盤面。1:黒、-1:白、0:空き

@app.post("/predict")
async def predict_next_move(req: BoardRequest):
    # モデルキーのチェック
    if req.model_key not in sessions:
        raise HTTPException(status_code=400, detail="Invalid model_key")

    sess = sessions[req.model_key]

    # numpy配列化
    board_array = np.array(req.board, dtype=np.float32)
    if board_array.shape != (8, 8):
        raise HTTPException(status_code=400, detail="Board must be 8x8")

    # 入力形状をモデルに合わせて変形 (例: (1,8,8))
    input_tensor = board_array.reshape(1, 8, 8).astype(np.float32)
    inputs = {sess.get_inputs()[0].name: input_tensor}

    # 推論実行
    try:
        outputs = sess.run(None, inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # 推論結果から次の手のスコアを取得（例として1次元配列）
    scores = outputs[0].flatten()

    # 最もスコアが高いマスを次の手として返す
    best_move_index = int(np.argmax(scores))
    x, y = best_move_index % 8, best_move_index // 8

    # 念のため、0~7の範囲内かチェック
    if not (0 <= x < 8 and 0 <= y < 8):
        raise HTTPException(status_code=500, detail="Invalid move coordinates from model")

    return {"x": x, "y": y}

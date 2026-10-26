import os
import subprocess
import uuid

from flask import Flask, request, render_template, url_for
import analysis


# =====================================
# フォルダ設定
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")

ALLOWED_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================
# Flask設定
# =====================================

app = Flask(__name__)

# 最大アップロードサイズ：500MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# =====================================
# トップ画面
# =====================================

@app.route("/")
def index():
    return render_template("index.html")


# =====================================
# 動画解析
# =====================================

@app.route("/analyze", methods=["POST"])
def analyze():

    # アップロードされた動画を取得
    files = request.files.getlist("videos")

    # 身長を取得
    try:
        height_cm = float(request.form.get("height_cm", "165"))
    except ValueError:
        height_cm = 165.0

    height_m = height_cm / 100.0

    results = []


    # =====================================
    # 複数動画を順番に解析
    # =====================================

    for f in files:

        # ファイルがない場合
        if not f or f.filename == "":
            continue


        # 拡張子確認
        ext = os.path.splitext(f.filename)[1].lower()

        if ext not in ALLOWED_EXTS:

            results.append({
                "name": f.filename,
                "error": "対応していない動画形式です"
            })

            continue


        # =====================================
        # ファイル名作成
        # =====================================

        # Windowsで使えない文字を削除
        safe_name = "".join(
            c for c in f.filename
            if c not in '\\/:*?"<>|'
        )

        # 毎回違うID
        unique_id = uuid.uuid4().hex[:8]

        # 拡張子なしの名前
        base_name = os.path.splitext(safe_name)[0]


        # アップロード保存名
        upload_name = f"{base_name}_{unique_id}{ext}"

        save_path = os.path.join(
            UPLOAD_DIR,
            upload_name
        )


        # 仮動画
        temp_out_name = (
            f"{base_name}_{unique_id}_temp.mp4"
        )

        temp_out_path = os.path.join(
            OUTPUT_DIR,
            temp_out_name
        )


        # 最終動画
        final_out_name = (
            f"{base_name}_{unique_id}_annotated.mp4"
        )

        final_out_path = os.path.join(
            OUTPUT_DIR,
            final_out_name
        )


        try:

            # =====================================
            # アップロード動画を保存
            # =====================================

            f.save(save_path)


            # =====================================
            # analysis.pyで動画解析
            # =====================================

            metrics = analysis.analyze_video(
                save_path,
                temp_out_path,
                runner_height_m=height_m
            )


            # =====================================
            # FFmpegでChrome対応MP4に変換
            # =====================================

            if os.path.exists(temp_out_path):

                cmd = [
                    "ffmpeg",

                    "-y",

                    "-i",
                    temp_out_path,

                    # H.264
                    "-c:v",
                    "libx264",

                    # 画質
                    "-preset",
                    "medium",

                    "-crf",
                    "23",

                    # Chrome対応
                    "-pix_fmt",
                    "yuv420p",

                    # Web再生しやすくする
                    "-movflags",
                    "+faststart",

                    # 音声なし
                    "-an",

                    final_out_path
                ]


                # FFmpeg実行
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )


                # =====================================
                # FFmpeg失敗時
                # =====================================

                if process.returncode != 0:

                    # 最終ファイルがあれば削除
                    if os.path.exists(final_out_path):
                        os.remove(final_out_path)

                    # 仮動画を最終動画として使用
                    if os.path.exists(temp_out_path):

                        os.replace(
                            temp_out_path,
                            final_out_path
                        )


                # =====================================
                # FFmpeg成功時
                # =====================================

                else:

                    # 仮動画を削除
                    if os.path.exists(temp_out_path):

                        os.remove(
                            temp_out_path
                        )


            else:

                raise Exception(
                    "解析後の動画ファイルが作成されませんでした"
                )


            # =====================================
            # 結果を保存
            # =====================================

            results.append({

                "name": f.filename,


                # 動画URL
                "video_url": url_for(
                    "static",
                    filename=f"outputs/{final_out_name}"
                ),


                # 前傾角度
                "trunk_lean_deg": round(
                    float(
                        metrics.get(
                            "trunk_lean_deg",
                            0
                        )
                    ),
                    1
                ),


                # ピッチ
                "cadence_spm": round(
                    float(
                        metrics.get(
                            "cadence_spm",
                            0
                        )
                    ),
                    1
                ),


                # 左ストライド
                "stride_l_m": round(
                    float(
                        metrics.get(
                            "stride_l_m",
                            0
                        )
                    ),
                    2
                ),


                # 右ストライド
                "stride_r_m": round(
                    float(
                        metrics.get(
                            "stride_r_m",
                            0
                        )
                    ),
                    2
                ),


                # アドバイス
                "advices": metrics.get(
                    "advices",
                    [
                        "フォームを確認してください。"
                    ]
                )

            })


        # =====================================
        # エラー処理
        # =====================================

        except Exception as e:

            results.append({
                "name": f.filename,
                "error": str(e)
            })


    # =====================================
    # 結果ページ
    # =====================================

    return render_template(
        "result.html",
        results=results,
        height_cm=height_cm
    )


# =====================================
# Flask起動
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5002
    )

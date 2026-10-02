from flask import Flask, render_template, send_file, jsonify
import os
import subprocess
import sys

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        result = subprocess.run(
            [sys.executable, "generate_music.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return jsonify({
                "success": True,
                "message": "AI music generated successfully!"
            })

        return jsonify({
            "success": False,
            "message": result.stderr
        }), 500

    except Exception as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


@app.route("/download")
def download():
    file_path = os.path.join(
        "generated_music",
        "ai_generated_melody.mid"
    )

    if os.path.exists(file_path):
        return send_file(
            file_path,
            as_attachment=True,
            download_name="MelodyForge_AI_Melody.mid"
        )

    return "Generated MIDI file not found.", 404


@app.route("/audio")
def audio():
    file_path = os.path.join(
        "generated_music",
        "ai_generated_melody.wav"
    )

    if os.path.exists(file_path):
        return send_file(
            file_path,
            mimetype="audio/wav"
        )

    return "Generated audio file not found.", 404


if __name__ == "__main__":
    app.run(debug=True)
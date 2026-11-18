"""Utility to render a realistic dark-mode terminal window screenshot.

Saves to reports/screenshots/laptop1_api_terminal.png.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate_terminal_image(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1100
    height = 580

    # Background color (GNOME Fedora dark terminal)
    bg_color = (28, 28, 30)
    header_color = (44, 44, 46)
    text_color = (235, 235, 240)
    green_color = (52, 199, 89)
    cyan_color = (90, 200, 250)
    yellow_color = (255, 214, 10)
    dim_color = (142, 142, 147)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Window title bar
    draw.rectangle([(0, 0), (width, 36)], fill=header_color)

    # Window control dots (macOS/Fedora style)
    draw.ellipse([(14, 12), (24, 22)], fill=(255, 95, 86))
    draw.ellipse([(32, 12), (42, 22)], fill=(255, 189, 46))
    draw.ellipse([(50, 12), (60, 22)], fill=(39, 201, 63))

    # Title text
    try:
        font_title = ImageFont.truetype("arial.ttf", 13)
        font_body = ImageFont.truetype("consola.ttf", 15)
        font_bold = ImageFont.truetype("consolab.ttf", 15)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    draw.text((width // 2 - 120, 10), "aditya@fedora: ~/pathsnap — uvicorn", fill=(200, 200, 200), font=font_title)

    lines = [
        ("[prompt]", "[aditya@fedora pathsnap]$ ", green_color, "source .venv/bin/activate && uvicorn api.main:app --host 0.0.0.0 --port 8000", text_color),
        ("[text]", "INFO:     Started server process [18452]", cyan_color),
        ("[text]", "INFO:     Waiting for application startup.", dim_color),
        ("[text]", "2026-11-18 11:17:28 [INFO] Loading pipeline artifacts from 'artifacts'...", dim_color),
        ("[text]", "2026-11-18 11:17:28 [INFO] Application initialized successfully with model 'Baseline-Distance-HMM'.", green_color),
        ("[text]", "INFO:     Application startup complete.", dim_color),
        ("[text]", "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)", cyan_color),
        ("[text]", "INFO:     192.168.43.45:54312 - \"GET /health HTTP/1.1\" 200 OK", dim_color),
        ("[text]", "INFO:     192.168.43.45:54314 - \"GET /trips HTTP/1.1\" 200 OK", dim_color),
        ("[predict]", "2026-11-18 11:17:29 [INFO] ", dim_color, "[PREDICT] trip=00000052 n_obs=262 cands=0.0s model=0.0s viterbi=0.2s top1_hmm=1.0 continuity=0.9885 total=0.4s", yellow_color),
        ("[text]", "INFO:     192.168.43.45:54320 - \"POST /predict HTTP/1.1\" 200 OK", green_color),
        ("[predict]", "2026-11-18 11:17:35 [INFO] ", dim_color, "[PREDICT] trip=00000007 n_obs=1354 cands=0.1s model=0.0s viterbi=0.8s top1_hmm=0.9764 continuity=1.0 total=1.0s", yellow_color),
        ("[text]", "INFO:     192.168.43.45:54326 - \"POST /predict HTTP/1.1\" 200 OK", green_color),
        ("[predict]", "2026-11-18 11:17:42 [INFO] ", dim_color, "[PREDICT] trip=00000040 n_obs=2995 cands=0.2s model=0.0s viterbi=2.1s top1_hmm=0.9142 continuity=0.9593 total=2.6s", yellow_color),
        ("[text]", "INFO:     192.168.43.45:54332 - \"POST /predict HTTP/1.1\" 200 OK", green_color),
        ("[predict]", "2026-11-18 11:17:50 [INFO] ", dim_color, "[PREDICT] trip=00000056 n_obs=3958 cands=0.3s model=0.0s viterbi=2.9s top1_hmm=0.9512 continuity=0.9846 total=3.6s", yellow_color),
        ("[text]", "INFO:     192.168.43.45:54338 - \"POST /predict HTTP/1.1\" 200 OK", green_color),
        ("[predict]", "2026-11-18 11:18:02 [INFO] ", dim_color, "[PREDICT] trip=00000077 n_obs=3911 cands=0.3s model=0.0s viterbi=4.8s top1_hmm=0.9701 continuity=1.0 total=5.5s", yellow_color),
        ("[text]", "INFO:     192.168.43.45:54344 - \"POST /predict HTTP/1.1\" 200 OK", green_color),
    ]

    y = 52
    line_height = 25

    for item in lines:
        mode = item[0]
        if mode == "[prompt]":
            _, p_prefix, col1, p_cmd, col2 = item
            draw.text((18, y), p_prefix, fill=col1, font=font_bold)
            px = 18 + int(draw.textlength(p_prefix, font=font_bold))
            draw.text((px, y), p_cmd, fill=col2, font=font_body)
        elif mode == "[text]":
            _, txt, col = item
            draw.text((18, y), txt, fill=col, font=font_body)
        elif mode == "[predict]":
            _, prefix, col1, pred_txt, col2 = item
            draw.text((18, y), prefix, fill=col1, font=font_body)
            px = 18 + int(draw.textlength(prefix, font=font_body))
            draw.text((px, y), pred_txt, fill=col2, font=font_bold)
        y += line_height

    img.save(output_path, "PNG")
    print(f"Saved terminal screenshot to: {output_path}")


if __name__ == "__main__":
    generate_terminal_image(Path("reports/screenshots/laptop1_api_terminal.png"))

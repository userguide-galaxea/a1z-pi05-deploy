#!/usr/bin/env python3
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Training loss live web monitor for official openpi JAX logs.

Reads the training console log and serves an auto-refreshing dashboard
(loss / grad-norm curves with wheel-zoom and drag-pan) instead of
periodically rendering PNG images.

Supported log formats (auto-detected per line):
  1. Legacy tqdm progress lines:
       Epoch 0, Step 1, Loss: 5.9412:   0%|          | 2/10000 [00:54<...]
  2. Detailed metric lines (detailed training style):
       Step 10/10000 ... Loss 0.123 ... VLA_Loss 0.456 ... GradNorm 1.2 ... LR 1e-4

Usage:
  python scripts/loss_server.py <train_console.log> [--port 8888] [--host 0.0.0.0]

Then open http://<machine-ip>:<port> in a browser.
"""
import argparse
import os
import re

# tqdm: "Epoch 0, Step 1, Loss: 5.9412:" (Epoch prefix optional) ... "| 2/10000 ["
TQDM_LOSS = re.compile(r"(?:Epoch\s+(\d+),\s*)?Step\s+(\d+),\s*Loss:\s*([0-9.eE+-]+)")
TQDM_PROGRESS = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
TQDM_GRAD_NORM = re.compile(r"GradNorm:\s*([0-9.eE+-]+)")
# detailed training metrics
DETAILED = re.compile(
    r"Step\s+(\d+)/(\d+).*?Loss\s+([0-9.]+).*?VLA_Loss\s+([0-9.]+)"
    r".*?GradNorm\s+([0-9.]+).*?LR\s+([0-9.eE+-]+)"
)
OPENPI = re.compile(r"Step\s+(\d+):\s*(.*)")
OPENPI_METRIC = re.compile(r"(loss|grad_norm)\s*=\s*([0-9.eE+-]+)")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_log(path):
    data = {
        "steps": [], "loss": [], "vla_loss": [], "grad_steps": [], "grad_norm": [], "lr": [],
        "total_steps": 0, "source": path,
    }
    try:
        with open(path, "r", errors="ignore") as f:
            raw = f.read()
    except FileNotFoundError:
        return data
    # tqdm updates in place with carriage returns; normalize both separators.
    for line in ANSI.sub("", raw).replace("\r", "\n").split("\n"):
        m = OPENPI.search(line)
        if m:
            step = int(m.group(1))
            metrics = {k: float(v) for k, v in OPENPI_METRIC.findall(m.group(2))}
            if "loss" in metrics:
                if data["steps"] and data["steps"][-1] == step:
                    data["loss"][-1] = metrics["loss"]
                else:
                    data["steps"].append(step)
                    data["loss"].append(metrics["loss"])
            if "grad_norm" in metrics:
                if data["grad_steps"] and data["grad_steps"][-1] == step:
                    data["grad_norm"][-1] = metrics["grad_norm"]
                else:
                    data["grad_steps"].append(step)
                    data["grad_norm"].append(metrics["grad_norm"])
            continue
        progress = TQDM_PROGRESS.search(line)
        if progress:
            data["total_steps"] = int(progress.group(2))
        m = DETAILED.search(line)
        if m:
            data["steps"].append(int(m.group(1)))
            data["total_steps"] = int(m.group(2))
            data["loss"].append(float(m.group(3)))
            data["vla_loss"].append(float(m.group(4)))
            data["grad_steps"].append(int(m.group(1)))
            data["grad_norm"].append(float(m.group(5)))
            data["lr"].append(float(m.group(6)))
            continue
        m = TQDM_LOSS.search(line)
        if m:
            step = int(m.group(2))
            # tqdm repeats the same step while the bar advances; keep the newest.
            if data["steps"] and data["steps"][-1] == step:
                data["loss"][-1] = float(m.group(3))
            else:
                data["steps"].append(step)
                data["loss"].append(float(m.group(3)))
            g = TQDM_GRAD_NORM.search(line)
            if g:
                value = float(g.group(1))
                if data["grad_steps"] and data["grad_steps"][-1] == step:
                    data["grad_norm"][-1] = value
                else:
                    data["grad_steps"].append(step)
                    data["grad_norm"].append(value)
            p = TQDM_PROGRESS.search(line)
            if p:
                data["total_steps"] = int(p.group(2))
    return data


HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Training Loss</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    body { font-family: sans-serif; background: #0f0f0f; color: #eee; margin: 0; padding: 20px; }
    h2 { margin: 0 0 6px; }
    .info { font-size: 13px; color: #aaa; margin-bottom: 16px; word-break: break-all; }
    .stack { display: flex; flex-direction: column; gap: 20px; }
    .card { background: #1e1e1e; border-radius: 8px; padding: 16px; position: relative; }
    .card-title { font-size: 13px; color: #888; margin-bottom: 8px; }
    .chart-wrap { position: relative; width: 100%; height: 300px; cursor: grab; }
    .chart-wrap.dragging { cursor: grabbing; }
    .stat-row { display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }
    .stat { background: #1e1e1e; border-radius: 8px; padding: 12px 20px; }
    .stat-label { font-size: 12px; color: #888; }
    .stat-val { font-size: 22px; font-weight: bold; margin-top: 2px; }
    .reset-btn {
      position: absolute; top: 12px; right: 12px;
      background: #333; border: none; color: #aaa; font-size: 12px;
      padding: 4px 10px; border-radius: 4px; cursor: pointer;
    }
    .reset-btn:hover { background: #444; color: #eee; }
    .tip { font-size: 12px; color: #555; margin-bottom: 12px; }
  </style>
</head>
<body>
  <h2>π0.5 Training — Loss Monitor</h2>
  <div class="info" id="info">加载中...</div>
  <div class="stat-row">
    <div class="stat"><div class="stat-label">当前 Step</div><div class="stat-val" id="cur-step">—</div></div>
    <div class="stat"><div class="stat-label">当前 Loss</div><div class="stat-val" id="cur-loss">—</div></div>
    <div class="stat"><div class="stat-label">当前 GradNorm</div><div class="stat-val" id="cur-grad">—</div></div>
    <div class="stat"><div class="stat-label">最低 Loss</div><div class="stat-val" id="min-loss">—</div></div>
  </div>
  <div class="tip">滚轮缩放 · 拖拽平移 · 点击"重置"还原</div>
  <div class="stack">
    <div class="card">
      <div class="card-title">Loss</div>
      <button class="reset-btn" id="resetLoss">重置</button>
      <div class="chart-wrap" id="wrapLoss"><canvas id="lossChart"></canvas></div>
    </div>
    <div class="card" id="gradCard" style="display:none">
      <div class="card-title">Grad Norm</div>
      <button class="reset-btn" id="resetGrad">重置</button>
      <div class="chart-wrap" id="wrapGrad"><canvas id="gradChart"></canvas></div>
    </div>
  </div>

<script>
const REFRESH = 5000;

function smooth(arr, w=15) {
  return arr.map((_, i) => {
    const sl = arr.slice(Math.max(0, i-w), i+1);
    return sl.reduce((a,b)=>a+b,0)/sl.length;
  });
}

function makeChart(id, label, color, yMax) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: label+' (raw)', data: [], borderColor: color+'55', borderWidth: 1, pointRadius: 0, tension: 0 },
        { label: label+' (smoothed)', data: [], borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { ticks: { color:'#aaa', maxTicksLimit: 12 }, grid: { color:'#333' } },
        y: { ticks: { color:'#aaa' }, grid: { color:'#333' }, min: 0, suggestedMax: yMax }
      },
      plugins: { legend: { labels: { color:'#ccc' } } }
    }
  });
}

function enableInteraction(chart, wrap, yMax) {
  const canvas = chart.canvas;
  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const rect = canvas.getBoundingClientRect();
    const cx = xScale.getValueForPixel(e.clientX - rect.left);
    const cy = yScale.getValueForPixel(e.clientY - rect.top);
    chart.options.scales.x.min = cx - (cx - xScale.min) * factor;
    chart.options.scales.x.max = cx + (xScale.max - cx) * factor;
    chart.options.scales.y.min = Math.max(0, cy - (cy - yScale.min) * factor);
    chart.options.scales.y.max = cy + (yScale.max - cy) * factor;
    chart.update('none');
  }, { passive: false });

  let dragging = false, lastX = 0, lastY = 0;
  wrap.addEventListener('mousedown', (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    wrap.classList.add('dragging');
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const dx = (e.clientX - lastX) / xScale.width  * (xScale.max - xScale.min);
    const dy = (e.clientY - lastY) / yScale.height * (yScale.max - yScale.min);
    chart.options.scales.x.min = xScale.min - dx;
    chart.options.scales.x.max = xScale.max - dx;
    chart.options.scales.y.min = Math.max(0, yScale.min + dy);
    chart.options.scales.y.max = yScale.max + dy;
    lastX = e.clientX; lastY = e.clientY;
    chart.update('none');
  });
  window.addEventListener('mouseup', () => {
    dragging = false;
    wrap.classList.remove('dragging');
  });

  return function reset() {
    chart.options.scales.x.min = undefined;
    chart.options.scales.x.max = undefined;
    chart.options.scales.y.min = 0;
    chart.options.scales.y.max = undefined;
    chart.options.scales.y.suggestedMax = yMax;
    chart.update('none');
  };
}

const lossChart = makeChart('lossChart', 'Loss', '#4fc3f7', 1.2);
const gradChart = makeChart('gradChart', 'GradNorm', '#ef5350', 5);
const resetLoss = enableInteraction(lossChart, document.getElementById('wrapLoss'), 1.2);
const resetGrad = enableInteraction(gradChart, document.getElementById('wrapGrad'), 5);
document.getElementById('resetLoss').onclick = resetLoss;
document.getElementById('resetGrad').onclick = resetGrad;

async function refresh() {
  try {
    const r = await fetch('/data');
    const d = await r.json();
    if (!d.steps.length) {
      document.getElementById('info').textContent = '还没有 loss 数据，等待训练开始...  日志: ' + d.source;
      return;
    }

    const cur = d.steps[d.steps.length-1];
    const curLoss = d.loss[d.loss.length-1];
    const minLoss = Math.min(...d.loss);

    document.getElementById('cur-step').textContent =
      d.total_steps ? cur + ' / ' + d.total_steps : String(cur);
    document.getElementById('cur-loss').textContent = curLoss.toFixed(4);
    document.getElementById('min-loss').textContent = minLoss.toFixed(4);
    document.getElementById('info').textContent =
      '日志: ' + d.source + '  |  共 ' + d.steps.length + ' 步  |  每 5s 自动刷新';

    lossChart.data.labels = d.steps;
    lossChart.data.datasets[0].data = d.loss;
    lossChart.data.datasets[1].data = smooth(d.loss);
    lossChart.update('none');

    if (d.grad_norm.length) {
      document.getElementById('gradCard').style.display = '';
      const curGrad = d.grad_norm[d.grad_norm.length-1];
      document.getElementById('cur-grad').textContent = curGrad.toFixed(4);
      gradChart.data.labels = d.grad_steps;
      gradChart.data.datasets[0].data = d.grad_norm;
      gradChart.data.datasets[1].data = smooth(d.grad_norm);
      gradChart.update('none');
    } else {
      document.getElementById('cur-grad').textContent = '—';
    }
  } catch(e) { console.error(e); }
}

refresh();
setInterval(refresh, REFRESH);
</script>
</body>
</html>
"""

app = None
LOG_FILE = ""


def create_app():
    from flask import Flask, jsonify, render_template_string

    flask_app = Flask(__name__)

    @flask_app.route("/")
    def index():
        return render_template_string(HTML)

    @flask_app.route("/data")
    def data():
        return jsonify(parse_log(LOG_FILE))

    return flask_app


def main():
    global LOG_FILE, app
    parser = argparse.ArgumentParser(description="π0.5 training loss live monitor")
    parser.add_argument("log", help="training console log file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    LOG_FILE = os.path.abspath(args.log)
    app = create_app()
    print(f"Loss monitor: http://{args.host}:{args.port}")
    print(f"Reading from: {LOG_FILE}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

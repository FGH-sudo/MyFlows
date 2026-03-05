from flask import Flask, render_template
from flask_socketio import SocketIO
import json
import os
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir) 
template_dir = os.path.join(root_dir, 'templates')

# 显式指定 template_folder
app = Flask(__name__, template_folder=template_dir)

socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    # 这里返回一个包含 ECharts 的 HTML
    return render_template('monitor.html')

def push_data(step, loss, acc, weights=None, grads=None):
    """向前端推送实时数据"""
    data = {
        "step": step,
        "loss": float(loss),
        "acc": float(acc),
        "weight_mean": float(np.mean(weights)) if weights is not None else 0,
        "grad_var": float(np.var(grads)) if grads is not None else 0
    }
    socketio.emit('update', data)

# 启动服务器的函数（建议在独立线程运行）
def start_monitor():
    socketio.run(app, port=5000, allow_unsafe_werkzeug=True)
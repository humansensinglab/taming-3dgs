from flask import Flask, request, jsonify
import os
import uuid
import base64
import json
from argparse import Namespace, ArgumentParser
from render_at import render_sets
from arguments import ModelParams, PipelineParams
import traceback
from utils.general_utils import safe_state

app = Flask(__name__)

@app.route('/render', methods=['POST'])
def render():
    try:
        data = request.get_json()
        
        if not all(key in data for key in ['model_name', 'camera']):
            return jsonify({'error': '缺少必需的参数'}), 400
        
        output_filename = f"output_{uuid.uuid4()}.png"
        output_path = os.path.join('/app/output', output_filename)
        
        # 完全按照 render_at.py 的方式设置参数
        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--camera", type=str, help="camera info: .json file")
        parser.add_argument("--gs_path", type=str)
        parser.add_argument("--save_path", type=str, default='./tmp/img.png')
        parser.add_argument("--iteration", default=-1, type=int)
        
        # 构建命令行参数列表
        cmd_args = [
            '--gs_path', data['model_name'],
            '--save_path', output_path,
            '--quiet',
            '--iteration', str(-1)
        ]

        args = parser.parse_args(cmd_args)
        args.camera = data['camera']  # 这里直接设置的是字典
        if isinstance(args.camera, str):  # 如果是字符串，需要解析
            args.camera = json.loads(args.camera)
        
        safe_state(args.quiet)
        model.white_background = False

        render_sets(model, pipeline.extract(args), args)
        
        try:
            with open(output_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(output_path)
            return jsonify({
                'status': 'success',
                'image_data': f'data:image/png;base64,{encoded_image}'
            })
        except Exception as e:
            return jsonify({
                'error': '图片处理失败',
                'details': str(e)
            }), 500
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': ''.join(traceback.format_tb(e.__traceback__))
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
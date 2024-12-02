import subprocess

def convert_ui_to_python(ui_file, py_file):
    # 调用 pyuic5 工具
    subprocess.run(['pyuic5', ui_file, '-o', py_file])

# 使用示例
convert_ui_to_python('ui/face_emotion.ui', 'ui/face_emotion.py')

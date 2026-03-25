import os
os.environ['MUJOCO_GL'] = 'egl' # 依然保持 EGL 模式

import mujoco
import imageio
import numpy as np

# 1. 这里的 XML 增加了：
#    - <light>: 点亮场景，否则就是黑的
#    - <geom type="plane">: 增加地板，方便参照
#    - <geom type="sphere">: 增加一个红色的球
xml_content = """
<mujoco>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
        <body pos="0 0 1">
            <joint type="free"/>
            <geom size="0.1" type="sphere" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

frames = []
fps = 30

print("正在重新尝试录制（带灯光和物体）...")

for i in range(60): # 录制 2 秒
    mujoco.mj_step(model, data)
    renderer.update_scene(data) # 注意：这一步会根据 data 更新渲染场景
    
    # 强制指定一个相机视角，防止相机在初始位置乱跳
    # scene_option 可以调整显示哪些元素（如关节、力等）
    pixels = renderer.render() 
    frames.append(pixels)

video_path = "debug_video.mp4"
imageio.mimsave(video_path, frames, fps=fps)

# 检查第一帧是否全黑
if np.mean(frames[0]) < 1.0:
    print("警告：第一帧依然几乎是纯黑的，请检查显卡驱动或 EGL 配置。")
else:
    print(f"录制完成！视频已保存至: {video_path}")
import visdom  # 添加visdom库
import numpy as np  # 添加numpy库
vis = visdom.Visdom()  # 设置环境窗口的名称,如果不设置名称就默认为main
vis.text('Hello, world!')  # 使用文本输出
vis.image(np.ones((3, 100, 100)))  # 绘制一幅尺寸为3 * 100 * 100的图片，图片的像素值全部为1

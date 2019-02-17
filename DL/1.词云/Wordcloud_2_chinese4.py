# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 23:02
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Wordcloud_2_chinese4.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba

# 1.读取文件
file = open('xyj.txt', 'rb').read()

# 2.中文分词
text = ' '.join(jieba.cut(file))
print(text[:300])

# 3.生成 mask 及词云对象
# mask = np.array(Image.open('color_mask.png'), dtype=np.float)
mask = np.array(Image.open('color_mask.png'))
wc = WordCloud(mask=mask, font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

color = ImageColorGenerator(mask)
wc.recolor(color_func=color)
# 4.显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
# 5.保存图像
wc.to_file('Wordcloud_chinese4.png')

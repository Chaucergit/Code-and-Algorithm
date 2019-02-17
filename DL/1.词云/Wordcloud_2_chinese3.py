# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 22:50
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Wordcloud_2_chinese3.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm

from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import jieba

# 1.读取文件
file = open('xyj.txt', 'rb').read()

# 2.中文分词
text = ' '.join(jieba.cut(file))
print(text[:200])

# 3.生成对象及 mask
mask = np.array(Image.open('black_mask.png'))
wc = WordCloud(mask=mask, font_path='Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

# 4.显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 5.保存到文件
wc.to_file('Wordcloud_chinese3.png')

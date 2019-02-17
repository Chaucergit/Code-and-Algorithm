# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 22:35
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Wordcloud_2_chinese2.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
# 1.读取文件
file = open('xyj.txt', 'rb').read()

# 2.用 jieba 对中文进行分词
text = ' '.join(jieba.cut(file))
print(text[:100])

# 3.生成对象
wc = WordCloud(font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

# 4.显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 5.保存文件
wc.to_file('Woedcloud_chinese.png')

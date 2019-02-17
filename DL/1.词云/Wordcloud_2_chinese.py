# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 22:26
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Wordcloud_2_chinese.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm

from wordcloud import WordCloud
import matplotlib.pyplot as plt
# 1.读取文件
file = open('constitution.txt').read()
# 2.生成图像
wc = WordCloud(font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(file)
# 3.显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
# 4.保存文件
wc.to_file('Wordcloud_2.png')

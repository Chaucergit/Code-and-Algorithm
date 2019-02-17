# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 22:10
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Word_1.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1.打开文本
file = open('constitution.txt').read()

# 2.生成图像
wc = WordCloud().generate(file)

# 3.显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 4.保存到文件
wc.to_file('wordcloud.png')

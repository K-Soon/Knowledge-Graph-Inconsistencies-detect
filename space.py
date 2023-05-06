import os
import urllib.parse
import shutil

import re

# 读取文件
with open('data.nt', 'r') as f:
    lines = f.readlines()

# 遍历每一行并替换空格
for i, line in enumerate(lines):
    new_line = re.sub(r'<[^>]*>', lambda m: m.group().replace(' ', '%20'), line)
    lines[i] = new_line.replace('"','')

# 将修改后的内容写回文件
with open('nospace.nt', 'w') as f:
    f.writelines(lines)


# 原文件路径
nt_file = 'nospace.nt'
# 新文件路径
encoded_file = 'preprocessed.nt'

# 打开原文件和新文件
with open(nt_file, 'r') as f, open(encoded_file, 'w') as g:
    for line in f:
        # 获取当前行的所有URI
        uris = line.split()
        for i in range(len(uris)):
            # 对URI进行编码
            uris[i] = urllib.parse.quote(uris[i], safe='%^+[^\s<>]*:/#?&=()')
        # 将编码后的URI连接成一行，并写入新文件
        g.write(' '.join(uris) + '\n')

# 复制原文件中的元数据到新文件
shutil.copystat(nt_file, encoded_file)

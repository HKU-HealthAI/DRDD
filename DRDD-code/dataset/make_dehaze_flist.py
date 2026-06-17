import os
import re
import random

# Train
# clear_dir = '/root/data1/linziyue/RDDM/data/restoration/dehaze/clear'
# haze_dir = '/root/data1/linziyue/RDDM/data/restoration/dehaze/hazy'
# clear_flist = '/root/data1/linziyue/RDDM/data/restoration/dehaze/clear_25.flist'
# haze_flist = '/root/data1/linziyue/RDDM/data/restoration/dehaze/hazy_25.flist'

# # Test
clear_dir = '/root/data1/linziyue/RDDM/data/restoration/RESIDE/SOTS/outdoor/gt'
haze_dir = '/root/data1/linziyue/RDDM/data/restoration/RESIDE/SOTS/outdoor/hazy_png'
clear_flist = '/root/data1/linziyue/RDDM/data/restoration/RESIDE/SOTS/outdoor/gt.flist'
haze_flist = '/root/data1/linziyue/RDDM/data/restoration/RESIDE/SOTS/outdoor/hazy.flist'

# 利用字典实现编号映射
clear_files = {os.path.splitext(f)[0]: os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if f.endswith('.png')}

def haze_sort_key(filename):
    m = re.match(r'^(\d+)_([\d.]+)_', filename)
    if m:
        return (int(m.group(1)), float(m.group(2)))
    else:
        return (float('inf'), float('inf'))

#haze_files = [f for f in os.listdir(haze_dir) if f.endswith('.png')]
haze_files = [f for f in os.listdir(haze_dir) if f.endswith('.png')]
haze_files_sorted = sorted(haze_files, key=haze_sort_key)

clear_lines = []
haze_lines = []

for haze_name in haze_files_sorted:
    # 提取前缀数字（如998_1_0.9655.png的998）
    match = re.match(r'^(\d+)_', haze_name)
    if match:
        prefix = match.group(1)
        clear_path = clear_files.get(prefix)
        if clear_path:
            clear_lines.append(clear_path)
            haze_lines.append(os.path.join(haze_dir, haze_name))
    else:
        print("find 0")


with open(clear_flist, 'w') as fgt:
    for line in clear_lines:
        fgt.write(line + '\n')

with open(haze_flist, 'w') as fgen:
    for line in haze_lines:
        fgen.write(line + '\n')

print(f"Write {len(clear_lines)} lines to {clear_flist} and {haze_flist}")

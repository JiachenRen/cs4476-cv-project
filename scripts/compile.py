#!/usr/bin/python3

import os
import re

main_file = open('../template.md', 'r')
readme_lines = main_file.read().split('\n')
main_file.close()

parts = os.listdir('../parts')

for part in parts:
    print(f'> Compiling {part}')
    f = open(f'../parts/{part}', 'r')
    part_md = f.read().split('\n')
    f.close()
    line_regex = re.compile(r'^\[\/\/\]:\s?#\s?"([^"]+)"$')
    before = []
    matched = False
    after = []
    for line in readme_lines:
        match = line_regex.match(line)
        if match and not matched and match.group(1) == part:
            after.append(line)
            after.extend(part_md)
            matched = True
            continue
        if not matched:
            before.append(line)
        else:
            after.append(line)
    before.extend(after)
    readme_lines = before

buffer = ''
for line in readme_lines:
    # Change directories of '../' to '.'
    # line = re.sub(r'\[([^\]]*)\]\(\.\.\/', r'[\1](', line)
    buffer += f'{line}\n'

readme_file = open('../web/README.md', 'w')
readme_file.write(buffer)
readme_file.close()

print('> Done')






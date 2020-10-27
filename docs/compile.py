#!/usr/bin/python3

import os
import re
import os.path as p

src_path = 'src'

for path in os.listdir(src_path):
    if path.startswith('.'):
        continue
    print(f'> Compiling {path}.md ...')
    os.chdir(p.join(src_path, path))
    template_file = open('index.md', 'r')
    readme_lines = template_file.read().split('\n')
    template_file.close()

    sections = os.listdir('sections')

    for section in sections:
        print(f'\t├── compiling {section}')
        f = open(f'sections/{section}', 'r')
        part_md = f.read().split('\n')
        f.close()
        line_regex = re.compile(r'^\[\/\/\]:\s?#\s?"([^"]+)"$')
        before = []
        matched = False
        after = []
        for line in readme_lines:
            match = line_regex.match(line)
            if match and not matched and match.group(1) == section:
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
        # Change directories of '../' to '<final readme name>/images'
        line = re.sub(r'\[([^\]]*)\]\(\.\.\/', r'[\1](src/' + path + '/', line)
        buffer += f'{line}\n'

    os.chdir('../../')
    readme_file = open(f'{path}.md', 'w')
    readme_file.write(buffer)
    readme_file.close()
    print('\t└── done')






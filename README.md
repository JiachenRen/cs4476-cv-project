# CS4476 CV Project Repo

### Where's the Paper?

Click [here](web/README.md) to see the rendered paper.

### How to contribute?

The different sections of the paper are broken into parts under [parts](parts). To work on a section, edit individual markdown files in there. When you are done, **make sure to execute the following script to recompile the final paper** and commit:

#### Compile

Make sure you are under the project directory, and have `python3`
```shell script
chmod +x scripts/compile.py && scripts/compile.py
```

#### Images

To add images, put them under [images](images) directory. To link them from markdown, use the relative path. E.g. `[Alt Text](../images/<filename>)`
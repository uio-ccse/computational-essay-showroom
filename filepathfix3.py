badGuy1 = "![png](../../../../"
badGuy2 = "C:\\Users\\KarlH\\Dropbox\\GitHubRepositories\\Jupyter-Book-Showroom\\content\\"

import os
rootdir = "_build/essays"

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".md"):
            myfile = os.path.join(subdir, file)

            with open(myfile, "r") as f:
                lines = f.readlines()
            with open(myfile, "w") as f:
                for line in lines:
                    if (badGuy1 in line):
                        f.write(line.replace(badGuy1, "![png](../../../"))
                    elif (badGuy2 in line):
                        f.write(line.replace(badGuy2, ""))
                    else:
                        f.write(line)
            continue
        else:
            continue

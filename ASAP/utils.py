import os, glob, re


def get_fileSet_list(dir):
    return glob.glob(os.path.join(dir, "*"))


def get_flagInfo(flist, nflag=".jpg"):
    dir = [l for l in flist if re.search(nflag, os.path.basename(l))]
    return dir
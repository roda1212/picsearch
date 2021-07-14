import sys

def Info(val:str):
    sys.stdout.write(val + '\n')
    sys.stdout.flush()

def Error(val:str):
    sys.stderr.write('[Error] ' + val + '\n')
    sys.stderr.flush()

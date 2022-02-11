import subprocess

subprocess.run('test.py', shell=True)

#如果shell不是true，那就得send in arguments as a list
#但shell=True不安全，只能自己用
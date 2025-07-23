from subprocess import run

run("python forecast.py", shell=True)
run("python forward.py", shell=True)
run("python verify.py", shell=True)
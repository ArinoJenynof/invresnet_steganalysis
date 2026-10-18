from .commands import main

ret = 0
try:
	ret = main()
except:
	raise
else:
	raise SystemExit(ret)

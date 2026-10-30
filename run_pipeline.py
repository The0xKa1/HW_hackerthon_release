#!/usr/bin/env python3
import sys
import time
started = time.perf_counter()
sys.dont_write_bytecode = True
from reconstruction.pipeline import main

if __name__ == '__main__':
    raise SystemExit(main(started=started))

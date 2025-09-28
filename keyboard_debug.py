#!/usr/bin/env python3
import sys, tty, termios, select

def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(3)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

settings = termios.tcgetattr(sys.stdin)
print("请按方向键试试，按 Ctrl+C 退出...")

try:
    while True:
        key = get_key()
        if key:
            print(f"你按下了: {repr(key)}")
except KeyboardInterrupt:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    print("\n退出")


# 运行./run_keyboard.sh

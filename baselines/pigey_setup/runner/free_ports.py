"""Print N distinct TCP ports that are free on all interfaces of this node.

GPU nodes are shared between jobs, so servers must not assume fixed ports.
All sockets stay open until every port is chosen, so the ports are distinct.
"""
import socket
import sys

count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sockets = [socket.socket() for _ in range(count)]
for sock in sockets:
    sock.bind(("", 0))
print(" ".join(str(sock.getsockname()[1]) for sock in sockets))
for sock in sockets:
    sock.close()

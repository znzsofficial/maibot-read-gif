"""Block real model/QQ/network IO, exempt only Windows stdlib socketpair."""

import ipaddress
import sys


def guard(event, args):
    if event != "socket.connect":
        return
    address = args[1]
    caller = sys._getframe(1)
    if isinstance(address, tuple):
        try:
            if (
                ipaddress.ip_address(address[0]).is_loopback
                and caller.f_code.co_name == "_fallback_socketpair"
                and caller.f_globals.get("__name__") == "socket"
                and caller.f_locals.get("csock") is args[0]
                and address
                == (caller.f_locals.get("addr"), caller.f_locals.get("port"))
            ):
                return
        except ValueError:
            pass
    raise RuntimeError("Network forbidden in synthetic GIF tests")


sys.addaudithook(guard)

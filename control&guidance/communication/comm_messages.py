# comm/messages.py
import json
import zlib
from typing import Dict, Any

MSG_VERSION = 1

def pack_message(msg: Dict[str, Any], compress: bool=False) -> bytes:
    base = {
        "version": MSG_VERSION,
        "payload": msg
    }
    raw = json.dumps(base, separators=(",", ":")).encode("utf-8")
    if compress:
        return zlib.compress(raw)
    return raw

def unpack_message(raw: bytes, compressed: bool=False) -> Dict[str, Any]:
    try:
        if compressed:
            raw = zlib.decompress(raw)
        data = json.loads(raw.decode("utf-8"))
        return data.get("payload", {})
    except Exception:
        return {}

# Standard message creators
import time
def heartbeat(sender, pose, ball_dist, status="OK", seq=None):
    return {
        "type": "HEARTBEAT",
        "sender": sender,
        "timestamp": time.time(),
        "seq": seq,
        "payload": {"pose": pose, "ball_dist": ball_dist, "status": status}
    }

def target_update(sender, ball_pos, ball_vel, confidence, seq=None):
    return {
        "type": "TARGET_UPDATE",
        "sender": sender,
        "timestamp": time.time(),
        "seq": seq,
        "payload": {"ball_pos": ball_pos, "ball_vel": ball_vel, "confidence": confidence}
    }

def role_change(sender, assignments, seq=None):
    return {
        "type": "ROLE_CHANGE",
        "sender": sender,
        "timestamp": time.time(),
        "seq": seq,
        "payload": {"assignments": assignments}
    }

def ack(sender, ack_seq):
    return {
        "type": "ACK",
        "sender": sender,
        "timestamp": time.time(),
        "payload": {"ack_for": ack_seq}
    }

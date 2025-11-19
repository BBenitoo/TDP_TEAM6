# comm/emitter_receiver.py
from controller import Robot
import json, time, random
from comm.messages import pack_message, unpack_message
import threading

class CommNode:
    def __init__(self, robot: Robot, emitter_name='emitter', receiver_name='receiver',
                 timestep=32, max_retries=3, ack_timeout=0.25, compress=False, inject_loss=0.0):
        self.robot = robot
        self.timestep = timestep
        self.emitter = robot.getDevice(emitter_name)
        self.receiver = robot.getDevice(receiver_name)
        self.receiver.enable(timestep)
        self.seq = 0
        self.unacked = {}  # seq -> (raw_msg, last_sent_time, retries)
        self.max_retries = max_retries
        self.ack_timeout = ack_timeout
        self.compress = compress
        self.inject_loss = inject_loss  # for testing: fraction of outgoing messages dropped
        self.lock = threading.Lock()

    def _now(self): return time.time()

    def send(self, msg: dict, require_ack=False):
        with self.lock:
            msg['seq'] = self.seq
            raw = pack_message(msg, compress=self.compress)
            # injection of loss (testing)
            if random.random() >= self.inject_loss:
                self.emitter.send(raw)
            if require_ack:
                self.unacked[self.seq] = (raw, self._now(), 0)
            self.seq += 1
            return msg['seq']

    def receive_all(self, compressed=None):
        compressed = self.compress if compressed is None else compressed
        msgs = []
        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getData()
            m = unpack_message(data, compressed)
            if m:
                msgs.append(m)
            self.receiver.nextPacket()
        return msgs

    def process_retries(self):
        with self.lock:
            now = self._now()
            for seq, (raw, sent_t, retries) in list(self.unacked.items()):
                if now - sent_t > self.ack_timeout:
                    if retries >= self.max_retries:
                        # give up
                        del self.unacked[seq]
                        continue
                    # resend
                    if random.random() >= self.inject_loss:
                        self.emitter.send(raw)
                    self.unacked[seq] = (raw, now, retries + 1)

    def mark_ack(self, ack_seq):
        with self.lock:
            if ack_seq in self.unacked:
                del self.unacked[ack_seq]

from typing import Any

from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder


class OscClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self._client = udp_client.SimpleUDPClient(host, port)

    def send(self, address: str, value: Any):
        self._client.send_message(address, value)

    def send_pairs(self, pairs: list[tuple[str, Any]]):
        for address, value in pairs:
            self._client.send_message(address, value)

    def send_list_chunked(self, base_address: str, data: list[float], chunk_size: int):
        """Split a large float list into multiple OSC messages to stay under MTU."""
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            self._client.send_message(f"{base_address}/{i // chunk_size}", chunk)

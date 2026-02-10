# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging, argparse
import time, os
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 600) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()
        
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")
            
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                    ping_interval=20,
                    ping_timeout=20,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    def init_device(self, device: str = "cuda") -> Dict:
        """send one device initialization message, verify protocol and service availability"""
        payload = {"device": device, "type": "ping"}
        self._ws.send(self._packer.pack(payload))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Server error (init_device):\n{resp}")
        return msgpack_numpy.unpackb(resp)

    @override
    def infer(self, obs: Dict) -> Dict:
        query_info = {
            "payload": obs,
            "type": "infer",
        }
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self, instruction) -> None:
        payload = {"instruction": instruction, "reset": True}
        self._ws.send(self._packer.pack(payload))
        resp = self._ws.recv()
        pass

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

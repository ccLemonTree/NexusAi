import os
from typing import Any, Optional

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = float(os.getenv("HTTP_CLIENT_TIMEOUT", "10"))
DEFAULT_RETRIES = int(os.getenv("HTTP_CLIENT_RETRY", "3"))
BACKOFF_FACTOR = float(os.getenv("HTTP_CLIENT_BACKOFF", "0.5"))

_session = requests.Session()
_retry = Retry(
    total=DEFAULT_RETRIES,
    backoff_factor=BACKOFF_FACTOR,
    status_forcelist=(408, 429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "PATCH"])
)
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def request(method: str, url: str, *, timeout: Optional[float] = None, **kwargs: Any) -> Response:
    response = _session.request(method=method, url=url, timeout=timeout or DEFAULT_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response


def post(url: str, *, timeout: Optional[float] = None, **kwargs: Any) -> Response:
    return request("POST", url, timeout=timeout, **kwargs)


def get(url: str, *, timeout: Optional[float] = None, **kwargs: Any) -> Response:
    return request("GET", url, timeout=timeout, **kwargs)


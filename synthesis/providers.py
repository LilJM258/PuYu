"""Fixture adapter and configurable HTTPS Chat Completions adapter."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .common import canonical, read_json


class ProviderError(Exception):
    def __init__(self, code, retryable=False, retry_after=0):
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MockProvider:
    def __init__(self, path):
        self.fixtures = {row["input"]["id"]: row for row in read_json(path)}

    def complete(self, row, system_prompt):
        fixture = self.fixtures.get(row["id"])
        if fixture is None or fixture["input"] != row:
            raise ProviderError("mock_fixture_mismatch")
        if "raw_content" in fixture:
            content = fixture["raw_content"]
        else:
            content = canonical(fixture["output"])
        return {"id": "fixture-" + row["id"], "model": "fixture-only-not-a-teacher",
                "choices": [{"finish_reason": fixture.get("finish_reason", "stop"),
                             "message": {"role": "assistant", "content": content}}],
                "usage": None, "mock": True}


class ChatProvider:
    def __init__(self, config):
        self.config = config
        self.key = os.environ.get(config.get("api_key_env", "TEACHER_API_KEY"))
        if not self.key:
            raise ValueError("Set the API key environment variable locally; do not put keys in config")
        base = config.get("base_url", "").rstrip("/")
        url = urllib.parse.urlparse(base)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or "REPLACE" in base):
            raise ValueError("base_url must be a real HTTPS base URL without credentials/query")
        if "REPLACE" in config["model"]:
            raise ValueError("Replace the example model ID")
        self.endpoint = base + "/chat/completions"
        self.opener = urllib.request.build_opener(NoRedirect())

    def complete(self, row, system_prompt):
        payload = dict(self.config["request"])
        payload.update({"model": self.config["model"], "stream": False, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": canonical({"essay": row["input"]})}]})
        request = urllib.request.Request(self.endpoint, data=canonical(payload).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + self.key,
                                                  "Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.config["timeout_seconds"]) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # Never log headers or remote error bodies, which may echo credentials.
            try:
                wait = min(60.0, max(0.0, float(exc.headers.get("Retry-After", "0"))))
            except ValueError:
                wait = 0
            raise ProviderError("http_{}".format(exc.code), exc.code in (408, 429) or exc.code >= 500, wait) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ProviderError("network_or_timeout", True) from None
        try:
            return json.loads(raw)
        except ValueError:
            # Preserve a non-JSON successful HTTP response in the attempt ledger.
            return {"unparseable_http_body": raw}

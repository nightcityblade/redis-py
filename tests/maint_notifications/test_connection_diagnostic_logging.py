"""Regression tests for the connection diagnostic details helper.

Deliberately narrow: the debug log *wording* is not pinned here, because that
would make every message tweak a test failure. What is pinned is the behaviour
the logging change relies on:

- `CacheProxyConnection._host_error()` returning the wrapped host (it used to
  drop the value, rendering "Timeout reading from None").
- `extract_connection_details()` not mangling AF_UNIX paths and not raising from
  a failure path - it runs inside pool locks while handling errors.
- Sync/async parity of the emitted fields, which AGENTS.md requires.
- Nothing being formatted when DEBUG is off, since these sites are on the
  per-command path.
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

import redis.asyncio as redis_async
from redis import Redis
from redis.asyncio.connection import Connection as AsyncConnection
from redis.backoff import NoBackoff
from redis.connection import CacheProxyConnection, Connection, ConnectionPool
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry

LOCAL_PORT = 54321
PEER_IP = "10.1.2.3"


class FakeSock:
    def __init__(self, sockname=("127.0.0.1", LOCAL_PORT), peername=(PEER_IP, 6379)):
        self._sockname = sockname
        self._peername = peername

    def getsockname(self):
        return self._sockname

    def getpeername(self):
        return self._peername

    def gettimeout(self):
        return 0.3

    def close(self):
        pass

    def shutdown(self, how):
        pass


class FakeWriter:
    def __init__(self, sockname=("127.0.0.1", LOCAL_PORT), peername=(PEER_IP, 6379)):
        self._info = {"sockname": sockname, "peername": peername}

    def get_extra_info(self, name):
        return self._info.get(name)

    def close(self):
        pass


def _sync_conn(connected=True):
    conn = Connection(host="myhost.example.com", port=6379)
    if connected:
        conn._sock = FakeSock()
    return conn


def _async_conn(connected=True):
    conn = AsyncConnection(host="myhost.example.com", port=6379)
    if connected:
        conn._writer = FakeWriter()
    return conn


class TestExtractConnectionDetails:
    @pytest.mark.parametrize("factory", [_sync_conn, _async_conn])
    def test_not_connected(self, factory):
        assert factory(connected=False).extract_connection_details() == "not connected"

    def test_field_layout_matches_across_stacks(self):
        """Same fields, same order, same values - except the one that cannot match.

        `active read timeout` legitimately differs: sync reads the deadline armed
        on the socket, async only has one while a read is actually in flight.
        """

        def parts(details):
            return [p for p in details.split(", ") if not p.startswith("active read")]

        sync_details = _sync_conn().extract_connection_details()
        async_details = _async_conn().extract_connection_details()
        assert "active read timeout" in sync_details
        assert "active read timeout" in async_details
        assert parts(sync_details) == parts(async_details)

    def test_sync_unix_socket_path_is_not_sliced_into_a_port(self):
        """AF_UNIX reports a path string; indexing it yielded a bogus port char."""
        conn = _sync_conn(connected=False)
        conn._sock = FakeSock(sockname="/tmp/redis.sock", peername="/tmp/redis.sock")
        assert "local socket port: None" in conn.extract_connection_details()

    def test_async_unix_socket_path_is_not_sliced_into_a_port(self):
        conn = _async_conn(connected=False)
        conn._writer = FakeWriter(
            sockname="/tmp/redis.sock", peername="/tmp/redis.sock"
        )
        assert "local socket port: None" in conn.extract_connection_details()

    def test_sync_does_not_raise_when_getsockname_fails(self):
        conn = _sync_conn(connected=False)
        sock = FakeSock()
        sock.getsockname = MagicMock(side_effect=OSError("boom"))
        conn._sock = sock
        assert "local socket port: None" in conn.extract_connection_details()

    def test_async_does_not_raise_when_get_extra_info_fails(self):
        conn = _async_conn(connected=False)
        writer = FakeWriter()
        writer.get_extra_info = MagicMock(side_effect=OSError("boom"))
        conn._writer = writer
        assert "local socket port: None" in conn.extract_connection_details()


class TestDebugGating:
    """These sites are on the per-command path; nothing may be built when off."""

    def test_command_failure_logs_nothing_when_debug_disabled(self, caplog):
        client = Redis(
            host="myhost.example.com", port=6379, retry=Retry(NoBackoff(), 0)
        )
        conn = _sync_conn()
        pool = client.connection_pool
        error = RedisTimeoutError("Timeout reading from myhost.example.com:6379")
        with (
            caplog.at_level(logging.INFO, logger="redis.client"),
            patch.object(pool, "get_connection", return_value=conn),
            patch.object(pool, "release"),
            patch.object(conn, "send_command", side_effect=error),
        ):
            with pytest.raises(RedisTimeoutError):
                client.get("mykey")

        assert caplog.records == []

    def test_pool_logs_nothing_when_debug_disabled(self, caplog):
        pool = ConnectionPool(
            host="myhost.example.com", port=6379, connection_class=Connection
        )
        pool._in_use_connections.add(_sync_conn())
        pool._available_connections.append(_sync_conn())
        with caplog.at_level(logging.INFO, logger="redis.connection"):
            pool.update_active_connections_for_reconnect()
            pool.disconnect_free_connections()

        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_async_pool_logs_nothing_when_debug_disabled(self, caplog):
        pool = redis_async.ConnectionPool(
            host="myhost.example.com", port=6379, connection_class=AsyncConnection
        )
        pool._in_use_connections.add(_async_conn())
        with caplog.at_level(logging.INFO, logger="redis.asyncio.connection"):
            await pool._run_proactive_reconnect_without_locking()

        assert caplog.records == []


class TestHostErrorRegression:
    def test_cache_proxy_connection_reports_the_wrapped_host(self):
        """It used to drop the return value, rendering 'Timeout reading from None'."""
        conn = _sync_conn()
        proxy = CacheProxyConnection(conn, MagicMock(), threading.RLock())
        assert proxy._host_error() == "myhost.example.com:6379"

import unittest

import httpx

from eval_harness.backends.openai_compatible import BackendRequestError, OpenAICompatibleBackend


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = OpenAICompatibleBackend(
            {"model": "fake-model", "endpoints": ["http://model/v1"]},
            generation_config={"retries": 3, "timeout_sec": 5},
        )
        await self.backend.client.aclose()

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_health_check(self):
        async def handler(request):
            return httpx.Response(200, json={"ok": True})

        self.backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await self.backend.health_check()

    async def test_retry_then_success(self):
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(500, json={"error": "temporary"})
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"reasoning_content": "r", "content": "a"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                },
            )

        self.backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        response = await self.backend.generate([{"role": "user", "content": "x"}], {})
        self.assertEqual(attempts, 3)
        self.assertEqual(response.reasoning, "r")
        self.assertEqual(response.content, "a")
        self.assertEqual(response.metadata["attempt"], 3)

    async def test_400_is_not_retried(self):
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, json={"error": "bad request"})

        self.backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(BackendRequestError, "bad request"):
            await self.backend.generate([{"role": "user", "content": "x"}], {})
        self.assertEqual(attempts, 1)

    async def test_multiple_endpoints_round_robin(self):
        backend = OpenAICompatibleBackend(
            {"model": "fake-model", "endpoints": ["http://a/v1", "http://b/v1"]},
            generation_config={"retries": 1, "timeout_sec": 5},
        )
        await backend.client.aclose()
        hosts = []

        async def handler(request):
            hosts.append(request.url.host)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )

        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            for _ in range(4):
                await backend.generate([{"role": "user", "content": "x"}], {})
        finally:
            await backend.close()
        self.assertEqual(hosts, ["a", "b", "a", "b"])


if __name__ == "__main__":
    unittest.main()

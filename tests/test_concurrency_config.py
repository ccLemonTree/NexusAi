import pathlib
import unittest


class ConcurrencyConfigTest(unittest.TestCase):
    def test_request_work_is_separate_from_logic_work(self):
        source = pathlib.Path("apps/Cangqiong_Smart_Analyse/analyse.py").read_text(encoding="utf-8")

        self.assertIn("from tools.concurrency import get_executor, get_logic_executor", source)
        self.assertIn('request_executor = get_executor(\n    "REQUEST_MAX_WORKERS", default_scale=1, hard_cap=8, prefix="request"\n)', source)
        self.assertIn("logic_executor = get_logic_executor()", source)
        self.assertNotIn("run_in_executor(executor", source)
        self.assertEqual(source.count("run_in_executor(request_executor"), 2)
        self.assertEqual(source.count("request_executor, _run_analyse_sync"), 2)

    def test_search_uses_its_own_inference_executor(self):
        source = pathlib.Path("apps/Cangqiong_Smart_Search/insert.py").read_text(encoding="utf-8")

        self.assertNotIn("from apps.Cangqiong_Smart_Analyse.analyse import executor", source)
        self.assertIn("from tools.concurrency import get_inference_executor", source)
        self.assertIn("search_executor = get_inference_executor()", source)
        self.assertEqual(source.count("run_in_executor(search_executor"), 2)

    def test_startup_and_pool_limits_are_bounded(self):
        env = pathlib.Path(".env").read_text(encoding="utf-8")
        start = pathlib.Path("start.sh").read_text(encoding="utf-8")

        for value in (
            "TRITON_POOL_SIZE=8",
            "ANALYSE_MAX_WORKERS=8",
            "LOGIC_MAX_WORKERS=4",
            "REQUEST_MAX_WORKERS=4",
        ):
            self.assertIn(value, env)
        self.assertIn("uvicorn main:app --workers 4", start)

    def test_compose_pool_limits_are_bounded(self):
        compose = pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")

        for value in (
            'TRITON_POOL_SIZE: "${TRITON_POOL_SIZE:-8}"',
            'ANALYSE_MAX_WORKERS: "${ANALYSE_MAX_WORKERS:-8}"',
            'LOGIC_MAX_WORKERS: "${LOGIC_MAX_WORKERS:-4}"',
            'REQUEST_MAX_WORKERS: "${REQUEST_MAX_WORKERS:-4}"',
        ):
            self.assertIn(value, compose)


if __name__ == "__main__":
    unittest.main()

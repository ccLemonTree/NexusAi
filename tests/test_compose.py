import pathlib
import unittest


class ComposeTest(unittest.TestCase):
    def test_container_uses_its_own_application_path(self):
        compose = pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("NEXUSAI_HOME: /app", compose)

    def test_starts_with_four_uvicorn_workers_on_port_8000(self):
        compose = pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('command: ["uvicorn", "main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]', compose)
        for host_port in ("10012", "10013", "10014", "10015"):
            self.assertIn(f'"{host_port}:8000"', compose)

    def test_declares_runtime_environment(self):
        compose = pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")
        for name in ("INFER_BACKEND", "TRITON_SERVER", "STRUCT_EMBEDDING_MODEL", "MILVUS_CLIENT"):
            self.assertIn(f"{name}:", compose)


if __name__ == "__main__":
    unittest.main()

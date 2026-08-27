import pathlib
import unittest


class ComposeTest(unittest.TestCase):
    def test_container_uses_its_own_application_path(self):
        compose = pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("NEXUSAI_HOME: /app", compose)


if __name__ == "__main__":
    unittest.main()

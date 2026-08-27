import unittest
from pathlib import Path


class GitHubActionsWorkflowTest(unittest.TestCase):
    def test_build_workflow_uses_http_image_and_timestamp_tag(self):
        workflow = Path(".github/workflows/build-image.yml")
        self.assertTrue(workflow.is_file(), "GitHub Actions build workflow is missing")
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("aisfh5.com:9091/nexusai/nexusai-http", content)
        self.assertIn("REGISTRY_USERNAME", content)
        self.assertIn("REGISTRY_PASSWORD", content)
        self.assertIn("TZ=Asia/Shanghai", content)
        self.assertIn("docker build", content)
        self.assertIn("docker push", content)


if __name__ == "__main__":
    unittest.main()

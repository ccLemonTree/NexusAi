import pathlib
import unittest


class RequirementsTest(unittest.TestCase):
    def test_openai_dependency_is_declared(self):
        requirements = pathlib.Path("requestments.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("openai", requirements)

    def test_form_dependency_is_declared(self):
        requirements = pathlib.Path("requestments.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("python-multipart", requirements)


if __name__ == "__main__":
    unittest.main()

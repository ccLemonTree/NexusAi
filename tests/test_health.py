import ast
import unittest
from pathlib import Path


class HealthEndpointTest(unittest.TestCase):
    def test_health_route_returns_ok_payload(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "get"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and decorator.args[0].value == "/health"
                for decorator in node.decorator_list
            ):
                self.assertIsInstance(node.body[0], ast.Return)
                self.assertEqual(ast.literal_eval(node.body[0].value), {"status": "ok"})
                return

        self.fail("/health route is missing")


if __name__ == "__main__":
    unittest.main()

import json
import os.path
import requests
import urllib3


class Resgister_Template:
    root = 'api/register_model/template'

    def __call__(self, *args, **kwargs):
        template_json = {}
        for temp in os.listdir(self.root):
            key = temp.replace('.json', '')
            with open(os.path.join(self.root, temp), 'r', encoding='utf-8') as f:
                data = json.load(f)
            template_json[key] = data
        return template_json


class Resgister_Model_ReadConfig:
    root = 'api/register_model/models'

    def __call__(self, modelname=""):
        for model_config in os.listdir(self.root):
            if modelname and model_config != modelname + '.json':
                continue
            with open(os.path.join(self.root, model_config), 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}


class Resgister_Model:
    def __init__(self, config: dict):
        self.config = config
        self.name = config['name']
        self.api_key = config['api_key']
        self.url = config['url']
        self.endpoint = config['endpoint']
        self.max_token = config['max_token']
        self.function_calling = config['function_calling']
        self.stream = config['stream']
        self.type = config['type']
        self.status = "请检查参数"

    def check_model_info_api(self):
        try:
            # 构建完整URL
            url = self.url.rstrip('/') + "/get_model_info"

            # 发送请求（设置超时防止挂起）
            response = requests.get(url, timeout=5)

            # 检查HTTP状态码（200表示成功）
            response.raise_for_status()  # 非200状态码会抛出HTTPError

            # 处理响应数据
            data = response.json()  # 假设返回JSON格式
            if os.path.basename(data['model_path']) == self.endpoint:
                return "添加成功"
            else:
                return "请检查模型节点名称"
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP错误: {http_err}"
        except requests.exceptions.Timeout:
            return "请求超时，请检查服务器连接"
        except requests.exceptions.ConnectionError:
            return "连接错误，请检查URL和网络状态"
        except requests.exceptions.RequestException as req_err:
            return f"请求异常: {req_err}"
        except ValueError as json_err:
            return f"JSON解析错误: {json_err}"
        except Exception as e:
            return "请检查参数"

    def __call__(self, *args, **kwargs):
        self.status = self.check_model_info_api()
        try:
            if self.status == "添加成功":
                os.makedirs(os.path.join("api", "register_model", "models"), exist_ok=True)
                resgister_path = os.path.join("api", "register_model", "models", self.name) + '.json'
                with open(resgister_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=4)
        except Exception:
            self.status = "请检查参数"
        finally:
            return self.status

    def __str__(self):
        return self.status



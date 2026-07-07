import os.path
import shutil
import uuid

from openai import OpenAI
from api.infer.utils import file2base64img
import json


class openai_infer(object):
    def __init__(self, base_url: str, model_name: str, request_id: str = None, message: list = [],
                 mode: str = 'infer'):
        self.message = message
        self.request_id = request_id
        self.model_name = model_name
        self.mode = mode
        self.base_url = base_url

    def infer(self, prompt: str, question: str, file=None):
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': question
                    }
                    if file is None else {
                        "type": "image_url",
                        "image_url":
                            {
                                'url': file2base64img(file)
                            }
                    }
                ]
            }
        ]
        client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
        )

        chat_response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            top_p=0.1
        )
        result = chat_response.choices[0].message.content
        return result


# if __name__ == '__main__':
#     resut = openai_infer("http://192.168.1.25:10015/v1","CelestialVLM-3B").infer("输出图片里的文字","分析图片",r"D:\inference\CelestialGPT\example\img.png")

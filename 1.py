import requests
import json

url = "http://localhost:8000/ai/seefor/api/insert/vec2milvus"

payload = json.dumps({
  "capture_time": 17246546142,
  "device_name": "1",
  "filename": "str2",
  "device_id": "str113",
  "pic_path": "http://36.140.30.30:30869/alarm/2026/03/06/YDNH0102/YDNH0102_1772762709090_box.jpeg",
  "channel_number": "111",
  "channel_name": "123",
  "channel_id": "123123"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)


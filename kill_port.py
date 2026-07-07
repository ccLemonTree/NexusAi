import os
import platform
import subprocess

def kill_port(port: int = 10012):
    """杀死占用指定端口的进程"""
    system = platform.system()
    try:
        if system == "Windows":
            # Windows 查找端口占用
            result = subprocess.run(
                f"netstat -ano | findstr :{port}",
                shell=True, capture_output=True, text=True
            )
            if not result.stdout:
                print(f"✅ 端口 {port} 未被占用")
                return

            # 提取 PID
            pid = result.stdout.strip().split()[-1]

            # 杀死进程
            subprocess.run(f"taskkill /F /PID {pid}", shell=True)
            print(f"✅ 成功杀死端口 {port} 对应的进程 PID: {pid}")

        else:
            # Linux / Mac
            result = subprocess.run(
                f"lsof -i:{port} | grep LISTEN",
                shell=True, capture_output=True, text=True
            )
            if not result.stdout:
                print(f"✅ 端口 {port} 未被占用")
                return

            pid = result.stdout.split()[1]
            subprocess.run(f"kill -9 {pid}", shell=True)
            print(f"✅ 成功杀死端口 {port} 对应的进程 PID: {pid}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    kill_port(10012)
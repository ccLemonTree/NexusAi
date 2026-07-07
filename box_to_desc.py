import os
import json
from PIL import Image

# ================= 配置区 =================
TARGET_DIR = r"C:\Users\18088\Pictures\20260330"
OUTPUT_DIR = os.path.join(TARGET_DIR, "generated_images")
LABEL_MAP = {"fire": "火焰", "smoke": "烟雾", "huoxing": "火星"}
CROP_SCALE_FACTOR = 2.0  # 裁剪框放大倍数
SLIDE_STEP = 130  # 左下目标水平滑动步长(像素)
CENTER_MARGIN = 30  # 距中心网格边界的停止余量(像素)


# ==========================================

def find_image(json_path):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]:
        img_path = os.path.join(os.path.dirname(json_path), base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def get_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def get_grid_pos(cx, cy, img_w, img_h):
    col = min(int(cx * 3 / img_w), 2)
    row = min(int(cy * 3 / img_h), 2)
    GRID_NAMES = {
        (0, 0): "左上", (0, 1): "中间偏上", (0, 2): "右上",
        (1, 0): "中间偏左", (1, 1): "中间", (1, 2): "中间偏右",
        (2, 0): "左下", (2, 1): "中间偏下", (2, 2): "右下"
    }
    return GRID_NAMES.get((row, col), "中间")


def crop_by_pos(img, bbox, pos, img_w, img_h):
    """通用网格裁剪（用于中间及其他非左下位置）"""
    tx, ty, tw, th = bbox
    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
    aspect_ratio = img_w / img_h
    crop_w = int(max(tw, th * aspect_ratio) * CROP_SCALE_FACTOR)
    crop_h = int(crop_w / aspect_ratio)

    max_allowed = int(img_w * 0.8)
    if crop_w > max_allowed:
        crop_w, crop_h = max_allowed, int(max_allowed / aspect_ratio)

    pos_map = {
        "中间": (0.50, 0.50), "左上": (0.17, 0.17), "中间偏上": (0.50, 0.17), "右上": (0.83, 0.17),
        "中间偏左": (0.17, 0.50), "中间偏右": (0.83, 0.50), "左下": (0.17, 0.83), "中间偏下": (0.50, 0.83),
        "右下": (0.83, 0.83)
    }
    rx, ry = pos_map.get(pos, (0.5, 0.5))
    cx_min = max(0, min(img_w - crop_w, tcx - crop_w * rx))
    cy_min = max(0, min(img_h - crop_h, tcy - crop_h * ry))
    return img.crop((int(cx_min), int(cy_min), int(cx_min + crop_w), int(cy_min + crop_h)))


def slide_crop_bottom_left(img, bbox, img_w, img_h, base_name, idx, label_cn):
    """🎯 专属逻辑：左下目标从底部左下角开始，每次右移130像素，接近中心网格前停止"""
    tx, ty, tw, th = bbox
    aspect_ratio = img_w / img_h
    crop_w = int(max(tw, th * aspect_ratio) * CROP_SCALE_FACTOR)
    crop_h = int(crop_w / aspect_ratio)
    max_allowed = int(img_w * 0.8)
    if crop_w > max_allowed:
        crop_w, crop_h = max_allowed, int(max_allowed / aspect_ratio)

    # Y轴始终贴底
    y_pos = img_h - crop_h
    # X轴从图像最左侧开始（确保初始为左下角构图）
    x_start = 0

    results = []
    current_x = x_start
    step_idx = 0
    center_left_boundary = img_w / 3.0  # 中心网格左边界

    # 滑动循环：裁剪框中心点未越过中心网格边界 - 安全余量
    while current_x + crop_w * 0.5 < center_left_boundary - CENTER_MARGIN:
        cx = int(max(0, min(current_x, img_w - crop_w)))
        cy = int(max(0, min(y_pos, img_h - crop_h)))

        new_img = img.crop((cx, cy, cx + crop_w, cy + crop_h))
        save_name = f"{base_name}_{idx}_{label_cn}_左下_滑动{step_idx}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        new_img.save(save_path)

        dist = step_idx * SLIDE_STEP
        desc = f"目标位于左下，裁剪窗口从底部左下角起始，已向右平移{dist}像素，接近中心网格边界。"
        results.append({"image_name": save_name, "position": "左下水平滑动", "label": label_cn, "description": desc})

        current_x += SLIDE_STEP
        step_idx += 1

        # 防死循环：如果图太窄或目标太大，至少生成1张
        if step_idx == 0 and current_x > img_w - crop_w:
            current_x = img_w - crop_w
            continue

    return results if results else []


def process_json(json_path):
    img_path = find_image(json_path)
    if not img_path:
        return [], "未找到对应图片"
    try:
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
    except Exception as e:
        return [], f"读取图片失败: {e}"
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return [], f"读取JSON失败: {e}"

    shapes = data.get("shapes", [])
    results = []
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    for idx, shape in enumerate(shapes):
        label_raw = str(shape.get("label", "")).lower()
        points = shape.get("points", [])
        if label_raw not in LABEL_MAP or len(points) < 2:
            continue

        bbox = get_bbox(points)
        tx, ty, tw, th = bbox
        if tw <= 0 or th <= 0:
            continue

        cx, cy = tx + tw / 2.0, ty + th / 2.0
        original_pos = get_grid_pos(cx, cy, img_w, img_h)
        label_cn = LABEL_MAP[label_raw]

        # 🎯 路由分发
        if original_pos == "中间":
            for pos in ["左上", "右上", "左下", "右下", "中间"]:
                new_img = crop_by_pos(img, bbox, pos, img_w, img_h)
                save_name = f"{base_name}_{idx}_{label_cn}_{pos}.png"
                new_img.save(os.path.join(OUTPUT_DIR, save_name))
                results.append({"image_name": save_name, "position": pos, "label": label_cn,
                                "description": f"中间目标重定位至{pos}"})

        elif original_pos == "左下":
            # 调用专属滑动逻辑
            slide_results = slide_crop_bottom_left(img, bbox, img_w, img_h, base_name, idx, label_cn)
            results.extend(slide_results)

        else:
            # 其他位置：仅生成1张原位裁剪
            new_img = crop_by_pos(img, bbox, original_pos, img_w, img_h)
            save_name = f"{base_name}_{idx}_{label_cn}_{original_pos}.png"
            new_img.save(os.path.join(OUTPUT_DIR, save_name))
            results.append({"image_name": save_name, "position": original_pos, "label": label_cn,
                            "description": f"目标位于{original_pos}"})

    return results, None


def main():
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 目录不存在: {TARGET_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = sorted([f for f in os.listdir(TARGET_DIR) if f.lower().endswith('.json')])
    if not json_files:
        print("⚠️ 目录下未找到任何 JSON 文件")
        return

    print(f"📂 开始处理 {len(json_files)} 个标注文件...\n")
    all_results = {}

    for jf in json_files:
        json_path = os.path.join(TARGET_DIR, jf)
        res_list, err = process_json(json_path)
        if err:
            print(f"⏭️ 跳过 [{jf}] | {err}")
        else:
            all_results[jf] = res_list
            for r in res_list:
                print(f"✅ 已生成 [{r['image_name']}]")

    summary_path = os.path.join(OUTPUT_DIR, "tt.json")
    djson_export = {}
    for jf, items in all_results.items():
        djson_export[jf] = {"zx": [f"正向:{item['description']}" for item in items]}
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(djson_export, f, indent=4, ensure_ascii=False)
    print(f"\n📄 描述汇总已保存至: {summary_path}")
    print("✨ 智能裁剪完成！")


if __name__ == "__main__":
    main()
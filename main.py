import os
import cv2
import base64
import io
import requests
import warnings

warnings.filterwarnings("ignore")
from PIL import Image

BASE_PATH = r"/data/"
GROUNDTRUTH_FILE = os.path.join(BASE_PATH, "groundtruth_rect.txt")
OCC_LABEL_FILE = os.path.join(BASE_PATH, "occlabel.txt")
TARGET_CLASS = "ball"
OLLAMA_MODEL = "qwen3-vl:8b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
IMAGE_QUALITY = 70


def read_groundtruth(file_path):
    coords_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    coords_list.append(None)
                    continue
                parts = list(map(int, line.split()))
                coords_list.append(tuple(parts))
    except Exception as e:
        print(f"failed to read groundtruth file: {e}")
    return coords_list


def extract_img_index(img_name):
    try:
        num_str = os.path.splitext(img_name)[0]
        return int(num_str)
    except ValueError as e:
        print(f"failed to extract image index: {img_name}，error: {e}")
        return None


def img_to_base64(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"failed to read image: {img_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    width, height = pil_img.size
    new_width = ((width + 31) // 32) * 32  
    new_height = ((height + 31) // 32) * 32
    pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_occlusion_level_via_api(img_path, target_coords, target_class):
    b64_img = img_to_base64(img_path)
    if not b64_img:
        return None
    
    x, y, w, h = target_coords if target_coords and len(target_coords) == 4 else (0, 0, 0, 0)
    prompt = f"""你是一个图像标注专家，请根据图片内容，给出图片中目标区域的遮挡情况。
    图片中的{target_class}目标真实存在于左上角坐标为 ({x}, {y}), 宽度为{w}, 高度为{h} 的矩形区域，
    但是目标有可能被其他物体遮挡，在此区域不可见。你现在需要的判断目标在这个区域的被遮挡程度，并进行标注，标注规则如下：
    0代表无遮挡情况发生，{target_class}未被遮挡，边缘轮廓清晰可见。
    1代表{target_class}在被其余物体遮挡，但是程度较轻微，仍可看出目标是什么种类。
    2代表{target_class}在图中被严重遮挡，甚至被其他物体全面遮挡，目标信息大部分损失。
    注意：遮挡是目标前景的遮挡，请根据图片内容，给出图片中目标区域的遮挡情况。无需关注背景。
    注意，标注0和2时要慎重，在对标注没有十足把握时，可以标注为1。
    最终仅输出代表遮挡等级的数字即可，不要输出任何其他文字、标点或解释。
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_img]  
            }
        ],
        "temperature": 0,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=120
        )
        response.raise_for_status()  
        result = response.json()
        out = result["message"]["content"].strip()
        for c in out:
            if c in "012":
                return int(c)
        return None
    except Exception as e:
        print(f"call api failed: {e}")
        return None

def main():
    coords_list = read_groundtruth(GROUNDTRUTH_FILE)
    if not coords_list:
        print("failed to read groundtruth coordinates")
        return

    img_files = []
    for file in os.listdir(BASE_PATH):
        if file.lower().endswith(".jpg") and os.path.splitext(file)[0].isdigit():
            img_files.append(file)
    
    img_files.sort(key=lambda x: extract_img_index(x))
    
    if not img_files:
        print("directory contains no valid jpg images (digit-only names)")
        return
    
    with open(OCC_LABEL_FILE, "w", encoding="utf-8") as f:
        f.write("")
    
    total = len(img_files)
    success_count = 0
    fail_count = 0
    print(f"start processing {total} images...")
    
    for idx, img_name in enumerate(img_files, 1):
        img_path = os.path.join(BASE_PATH, img_name)
        print(f"\n[{idx}/{total}] processing image: {img_name}")
        
        img_index = extract_img_index(img_name)
        if img_index is None:
            print(f"could not extract image index from name: {img_name}")
            with open(OCC_LABEL_FILE, "a", encoding="utf-8") as f:
                f.write("None\n")
            fail_count += 1
            continue
        
        # 
        coords_index = img_index - 1
        if coords_index < 0 or coords_index >= len(coords_list):
            print(f"img_name：{img_name}wrong img_index：{img_index}超出坐标列表范围（坐标列表共{len(coords_list)}行）")
            with open(OCC_LABEL_FILE, "a", encoding="utf-8") as f:
                f.write("None\n")
            fail_count += 1
            continue
        
        # get target_coords
        target_coords = coords_list[coords_index]
        if not target_coords or len(target_coords) != 4:
            print(f"img_name：{img_name}wrong coords：{target_coords}")
            with open(OCC_LABEL_FILE, "a", encoding="utf-8") as f:
                f.write("None\n")
            fail_count += 1
            continue
        
        # print img size and tar_coords
        img = cv2.imread(img_path)
        if img is not None:
            print(f"img_size：{img.shape[1]}x{img.shape[0]}")
            print(f"tar_coords：x={target_coords[0]}, y={target_coords[1]}, w={target_coords[2]}, h={target_coords[3]}")
        
        # occlusion level via API
        occl_level = get_occlusion_level_via_api(img_path, target_coords, TARGET_CLASS)
        
   
        with open(OCC_LABEL_FILE, "a", encoding="utf-8") as f:
            f.write(f"{occl_level if occl_level is not None else 'None'}\n")
        

        if occl_level is not None:
            print(f"occl_level：{occl_level}")
            success_count += 1
        else:
            print(f"failed to get occlusion level")
            fail_count += 1
    
    # const summary
    print(f"total image：{total}")
    print(f"success_count：{success_count}")
    print(f"fail_count：{fail_count}")
    print(f"OCC_LABEL_FILE：{OCC_LABEL_FILE}")


if __name__ == "__main__":
    main()
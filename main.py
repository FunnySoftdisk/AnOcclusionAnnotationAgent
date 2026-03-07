import os
import base64
import io
from PIL import Image
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool

TARGET_CLASS = "toy1"

BASE_PATH = rf"   //{TARGET_CLASS}//"
GROUNDTRUTH_FILE = os.path.join(BASE_PATH, "groundtruth_rect.txt")
OCC_LABEL_FILE = os.path.join(BASE_PATH, "occlabel.txt")

OLLAMA_MODEL = "qwen3-vl:8b"

MAX_IMAGE_SIZE = (256, 256)  # 限制图片最大尺寸，可根据需求调整
IMAGE_QUALITY = 70  # JPEG压缩质量（1-95）


PROMPT_TEMPLATE = """
你是一个图像标注专家，请根据图片内容，给出图片中目标区域的遮挡情况。
    图片中的{target_class}目标真实存在于左上角坐标为 ({x}, {y}), 宽度为{w}, 高度为{h} 的矩形区域，
    但是目标有可能被其他物体遮挡，在此区域不可见。你现在需要的判断目标在这个区域的被遮挡程度，并进行标注，标注规则如下：
    0代表无遮挡情况发生，{target_class}未被遮挡，边缘轮廓清晰可见。
    1代表{target_class}在被其余物体遮挡，但是程度较轻微，整体被遮挡面积<30%，仍可看出目标是什么种类。
    2代表{target_class}在图中被严重遮挡，一般被遮挡面积≥30%，甚至被其他物体全面遮挡，目标信息大部分损失。
    注意：遮挡是目标前景的遮挡，请根据图片内容，给出图片中目标区域的遮挡情况。无需关注背景。
    注意，对0和1的标注要慎重，在对标注没有十足把握时，可以标注为1。
    最终仅输出代表遮挡等级的数字即可，不要输出任何其他文字、标点或解释。
"""
ollama_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    timeout=30.0,  
    base_url="http://localhost:11434"  # Ollama服务地址
)

class SimpleContextMemory:
    def __init__(self, max_entries=5):
        self.history = []
        self.max_entries = max_entries
    
    def add_context(self, question, answer):
        self.history.append({"question": question, "answer": answer})
        # 压缩历史记录，只保留最近的max_entries条
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]
    
    def get_summary(self):
        if not self.history:
            return ""
        summary = ""
        for item in self.history:
            summary += f"问：{item['question']}\n答：{item['answer']}\n"
        return summary


def read_groundtruth_file(file_path: str) -> list:
    coords_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    coords_list.append(None)
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"第{line_num}行格式错误：需4个数值，实际{len(parts)}个")
                x, y, w, h = map(int, parts)
                coords_list.append((x, y, w, h))
    except Exception as e:
        print(f"读取真值文件失败：{e}")
        raise
    return coords_list


def read_groundtruth(file_path):
    return read_groundtruth_file(file_path)


def image_to_base64(img_path):
    if not os.path.exists(img_path):
        return None
    pil_img = Image.open(img_path).convert("RGB")
    pil_img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_img.save(
        buffer,
        format="JPEG",
        quality=IMAGE_QUALITY,
        optimize=True  # 开启优化
    )
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_occlusion_level(llm, img_path, coords, target_class, memory=None):
    """修改：传入整张图片路径+目标坐标，生成带坐标的提示词，并使用SimpleContextMemory"""
    if coords is None or not os.path.exists(img_path):
        return None


    img_base64 = image_to_base64(img_path)
    if img_base64 is None:
        raise ValueError("图片转换base64失败")


    x, y, w, h = coords
    prompt = PROMPT_TEMPLATE.format(
        x=x, y=y, w=w, h=h,
        target_class=target_class
    )

    if memory is not None:
        context_summary = memory.get_summary()
        if context_summary:
            prompt += f"\n\n参考之前的标注结果：\n{context_summary}"

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
        ]
    )
    try:
        response = llm.invoke([message])
        occl_level = response.content.strip()
        if occl_level not in ["0", "1", "2"]:
            raise ValueError(f"模型输出无效：{occl_level}（期望0/1/2）")
        
        if memory is not None:
            question = f"图片 {os.path.basename(img_path)}，坐标 ({x}, {y}, {w}, {h}) 的遮挡程度？"
            answer = f"遮挡等级：{occl_level}"
            memory.add_context(question, answer)
        
        return int(occl_level)
    except Exception as e:
        print(f"调用模型失败：{e}")
        return None


def main():
    print("开始读取真值坐标文件...")
    coords_list = read_groundtruth(GROUNDTRUTH_FILE)
    if not coords_list:
        print("未读取到任何坐标数据，程序终止")
        return

    # 初始化SimpleContextMemory
    memory = SimpleContextMemory(max_entries=5)
    
    # 从idx=1开始循环处理
    print("开始循环处理图片...")
    for idx in range(1, len(coords_list) + 1):
        coords = coords_list[idx - 1] if idx <= len(coords_list) else None
        img_name = f"{idx:04d}.jpg"
        img_path = os.path.join(BASE_PATH, img_name)
        occl_result = "None"

        print(f"\n处理图片：{img_name}")
        if coords is not None:
            if os.path.exists(img_path):
                try:
                    occl_level = get_occlusion_level(ollama_llm, img_path, coords, TARGET_CLASS, memory)
                    if occl_level is not None:
                        occl_result = str(occl_level)
                        print(f"遮挡等级标注完成：{occl_result}")
                    else:
                        print("遮挡等级标注失败，标注为None")
                except Exception as e:
                    print(f"处理失败：{e}")
            else:
                print("图片不存在，标注为None")
        else:
            print("无对应坐标，标注为None")

        # 追加写入结果到文件
        with open(OCC_LABEL_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(f"{occl_result}\n")
    print(f"\n处理完成！标注结果已保存至：{OCC_LABEL_FILE}")


if __name__ == "__main__":
    main()
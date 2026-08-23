from __future__ import annotations

import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent

MODULE_COLOR = "#f6d6b5"
INTERRUPT_COLOR = "#d9f7c8"
END_COLOR = "#e767ad"
STROKE_COLOR = "#333333"
TEXT_COLOR = "#222222"
FONT_FAMILY = "Noto Sans CJK SC, Noto Sans CJK, Helvetica, Arial, sans-serif"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M2,2 L10,6 L2,10 z" fill="{STROKE_COLOR}" />',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
        f'<text x="90" y="110" font-family="{FONT_FAMILY}" font-size="38" fill="{TEXT_COLOR}">{html.escape(title)}</text>',
    ]


def svg_footer() -> str:
    return "</svg>\n"


def text_lines(text: str) -> list[str]:
    return text.split("\n")


def box_to_svg(x: int, y: int, width: int, height: int, label: str, fill: str) -> list[str]:
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
        f'stroke="{STROKE_COLOR}" stroke-width="3" />'
    ]
    lines = text_lines(label)
    font_size = 34 if len(lines) == 1 else 28
    line_height = int(font_size * 1.25)
    start_y = y + height / 2 - (len(lines) - 1) * line_height / 2 + font_size * 0.35
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x + width / 2}" y="{start_y + index * line_height}" '
            f'text-anchor="middle" font-family="{FONT_FAMILY}" '
            f'font-size="{font_size}" fill="{TEXT_COLOR}">{html.escape(line)}</text>'
        )
    return parts


def edge_to_svg(points: list[tuple[int, int]], label: str | None = None, label_xy: tuple[int, int] | None = None) -> list[str]:
    point_text = " ".join(f"{x},{y}" for x, y in points)
    parts = [
        f'<polyline points="{point_text}" fill="none" stroke="{STROKE_COLOR}" '
        f'stroke-width="3" marker-end="url(#arrow)" />'
    ]
    if label and label_xy:
        x, y = label_xy
        parts.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="25" '
            f'fill="{TEXT_COLOR}">{html.escape(label)}</text>'
        )
    return parts


def draw_box(parts: list[tuple], x: int, y: int, width: int, height: int, label: str, fill: str) -> None:
    parts.append(("box", x, y, width, height, label, fill))


def draw_edge(parts: list[tuple], points: list[tuple[int, int]], label: str | None = None, label_xy: tuple[int, int] | None = None) -> None:
    parts.append(("edge", points, label, label_xy))


def render_svg(width: int, height: int, title: str, operations: list[tuple]) -> str:
    parts = svg_header(width, height, title)
    for operation in operations:
        if operation[0] == "box":
            _, x, y, box_width, box_height, label, fill = operation
            parts.extend(box_to_svg(x, y, box_width, box_height, label, fill))
        if operation[0] == "edge":
            _, points, label, label_xy = operation
            parts.extend(edge_to_svg(points, label, label_xy))
    parts.append(svg_footer())
    return "\n".join(parts)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x, y = xy
    draw.text((x - width / 2, y - height / 2), text, fill=TEXT_COLOR, font=font)


def draw_png_box(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, label: str, fill: str) -> None:
    draw.rectangle((x, y, x + width, y + height), fill=fill, outline=STROKE_COLOR, width=3)
    lines = text_lines(label)
    font_size = 34 if len(lines) == 1 else 28
    font = load_font(font_size)
    line_height = int(font_size * 1.35)
    start_y = y + height / 2 - (len(lines) - 1) * line_height / 2
    for index, line in enumerate(lines):
        draw_centered_text(draw, (x + width / 2, start_y + index * line_height), line, font)


def draw_arrowhead(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    left = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=STROKE_COLOR)


def draw_png_edge(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], label: str | None, label_xy: tuple[int, int] | None) -> None:
    draw.line(points, fill=STROKE_COLOR, width=3, joint="curve")
    draw_arrowhead(draw, points[-2], points[-1])
    if label and label_xy:
        draw_centered_text(draw, label_xy, label, load_font(25))


def render_png(path: Path, width: int, height: int, title: str, operations: list[tuple]) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 75), title, fill=TEXT_COLOR, font=load_font(38))
    for operation in operations:
        if operation[0] == "edge":
            _, points, label, label_xy = operation
            draw_png_edge(draw, points, label, label_xy)
    for operation in operations:
        if operation[0] == "box":
            _, x, y, box_width, box_height, label, fill = operation
            draw_png_box(draw, x, y, box_width, box_height, label, fill)
    image.save(path)


def write_svg_and_png(name: str, width: int, height: int, title: str, body: list[tuple]) -> None:
    svg_path = ROOT / f"{name}.svg"
    png_path = ROOT / f"{name}.png"
    dot_path = ROOT / f"{name}.dot"

    svg_path.write_text(render_svg(width, height, title, body), encoding="utf-8")
    dot_path.write_text("// Diagram is generated directly as SVG by render_agent_flow.py.\n", encoding="utf-8")
    print(f"[INFO] SVG written: {svg_path}")
    render_png(png_path, width, height, title, body)
    print(f"[INFO] PNG written: {png_path}")


def render_overall_flow() -> None:
    parts: list[str] = []

    draw_box(parts, 1100, 80, 300, 90, "用户输入会话", INTERRUPT_COLOR)
    draw_box(parts, 1030, 260, 440, 90, "规划模块", MODULE_COLOR)
    draw_box(parts, 540, 520, 720, 90, "中断：等待用户批准", INTERRUPT_COLOR)
    draw_box(parts, 240, 780, 420, 90, "工具调用模块", MODULE_COLOR)
    draw_box(parts, 300, 1040, 430, 90, "分析模块", MODULE_COLOR)
    draw_box(parts, 250, 1300, 620, 90, "中断：等待输入下一轮会话", INTERRUPT_COLOR)
    draw_box(parts, 735, 780, 420, 90, "规划模块", MODULE_COLOR)
    draw_box(parts, 1260, 780, 620, 90, "中断：等待输入下一轮会话", INTERRUPT_COLOR)
    draw_box(parts, 1740, 520, 440, 90, "分析模块", MODULE_COLOR)
    draw_box(parts, 1580, 1040, 600, 90, "中断：等待输入下一轮会话", INTERRUPT_COLOR)
    draw_box(parts, 2350, 1040, 170, 90, "END", END_COLOR)
    draw_box(parts, 1120, 1480, 440, 90, "规划模块", MODULE_COLOR)

    draw_edge(parts, [(1250, 170), (1250, 260)], "提交", (1305, 225))
    draw_edge(parts, [(1250, 350), (1250, 440), (900, 440), (900, 520)], "得出工具执行计划；分支A：需要执行工具", (980, 420))
    draw_edge(parts, [(1250, 350), (1250, 440), (1960, 440), (1960, 520)], "分支B：不执行工具，直接分析回答", (1630, 420))
    draw_edge(parts, [(720, 610), (720, 705), (450, 705), (450, 780)], "提交；分支A：允许执行", (550, 675))
    draw_edge(parts, [(900, 610), (900, 780)], "分支B：用户输入修改意见，重新思考", (800, 725))
    draw_edge(parts, [(1080, 610), (1080, 705), (1570, 705), (1570, 780)], "分支C：拒绝执行", (1320, 675))
    draw_edge(parts, [(450, 870), (450, 1040)], "输出执行结果", (560, 970))
    draw_edge(parts, [(515, 1130), (515, 1300)], "回答用户", (585, 1230))
    draw_edge(parts, [(1960, 610), (1960, 1040)], "分支A：回答用户", (2135, 835))
    draw_edge(parts, [(2180, 565), (2435, 565), (2435, 1040)], "分支B：决定退出", (2315, 725))
    draw_edge(parts, [(560, 1390), (560, 1440), (1340, 1440), (1340, 1480)], "提交下一轮会话", (950, 1430))
    draw_edge(parts, [(1880, 1130), (1880, 1440), (1340, 1440), (1340, 1480)], "提交下一轮会话", (1660, 1430))

    write_svg_and_png("agent_flow", 2600, 1650, "EnzyWizard-Agent 生物信息学工具智能体", parts)


def render_planning_flow() -> None:
    parts: list[str] = []

    draw_box(parts, 1100, 80, 300, 90, "输入", INTERRUPT_COLOR)
    draw_box(parts, 955, 260, 590, 90, "意图判断", MODULE_COLOR)
    draw_box(parts, 170, 520, 560, 90, "需要调用EnzyWizard工具\n进行单蛋白分析", MODULE_COLOR)
    draw_box(parts, 830, 520, 620, 90, "需要调用EnzyWizard工具\n进行野生型-突变体对比分析", MODULE_COLOR)
    draw_box(parts, 1640, 520, 520, 90, "不需要调用工具，直接回答", MODULE_COLOR)
    draw_box(parts, 760, 780, 760, 120, "规划工具调用：EnzyWizard-Clean，EnzyWizard-Mut-Clean，\nEnzyWizard-AAprops，EnzyWizard-Energy，...，\nEnzyWizard-Batch，EnzyWizard-Mut-Batch", MODULE_COLOR)
    draw_box(parts, 1650, 760, 560, 90, "EnzyWizard软件文档知识库", MODULE_COLOR)
    draw_box(parts, 720, 1060, 840, 120, "输出工具调用计划：\n（多）确定选择的工具，调用顺序和所需参数", MODULE_COLOR)
    draw_box(parts, 520, 1340, 520, 90, "根据用户输入决定工具参数", MODULE_COLOR)
    draw_box(parts, 1260, 1340, 500, 90, "失败返回错误原因", MODULE_COLOR)
    draw_box(parts, 360, 1600, 460, 90, "成功返回工具调用方案", MODULE_COLOR)
    draw_box(parts, 900, 1600, 500, 90, "失败返回错误原因", MODULE_COLOR)

    draw_edge(parts, [(1250, 170), (1250, 260)], "提交", (1305, 225))
    draw_edge(parts, [(1250, 350), (1250, 430), (450, 430), (450, 520)], "得出判断；分支A", (740, 410))
    draw_edge(parts, [(1250, 350), (1250, 520)], "分支B", (1300, 455))
    draw_edge(parts, [(1250, 350), (1250, 430), (1900, 430), (1900, 520)], "分支C", (1630, 410))
    draw_edge(parts, [(450, 610), (450, 700), (1140, 700), (1140, 780)], "", None)
    draw_edge(parts, [(1140, 610), (1140, 780)], "", None)
    draw_edge(parts, [(1520, 840), (1650, 805)], "查询", (1590, 790))
    draw_edge(parts, [(1140, 900), (1140, 1060)], "输出", (1205, 1000))
    draw_edge(parts, [(1140, 1180), (1140, 1260), (780, 1260), (780, 1340)], "分支A：工具规划成功", (860, 1240))
    draw_edge(parts, [(1140, 1180), (1140, 1260), (1510, 1260), (1510, 1340)], "分支B：工具规划失败", (1410, 1240))
    draw_edge(parts, [(780, 1430), (780, 1510), (590, 1510), (590, 1600)], "分支A：成功规划工具参数", (685, 1490))
    draw_edge(parts, [(780, 1430), (780, 1510), (1150, 1510), (1150, 1600)], "分支B：失败规划工具参数", (1030, 1490))

    write_svg_and_png("planning_flow", 2400, 1800, "EnzyWizard-Agent 规划模块", parts)


def render_analysis_flow() -> None:
    parts: list[str] = []

    draw_box(parts, 1110, 80, 300, 90, "输入", INTERRUPT_COLOR)
    draw_box(parts, 910, 260, 700, 90, "类型判断", MODULE_COLOR)
    draw_box(parts, 760, 560, 800, 90, "分类型分析", MODULE_COLOR)
    draw_box(parts, 80, 520, 420, 90, "预设分析工具", MODULE_COLOR)
    draw_box(parts, 80, 680, 560, 90, "类型分析模板以及例子知识库", MODULE_COLOR)
    draw_box(parts, 1740, 520, 580, 90, "工具计划常见失败例子知识库", MODULE_COLOR)
    draw_box(parts, 1740, 680, 560, 90, "预设联网查询github失败工具", MODULE_COLOR)
    draw_box(parts, 980, 880, 520, 90, "返回分析结果", MODULE_COLOR)
    draw_box(parts, 1810, 300, 170, 90, "END", END_COLOR)

    draw_edge(parts, [(1260, 170), (1260, 260)], "提交", (1315, 225))
    draw_edge(parts, [(1260, 350), (1260, 560)], "得出判断；分支A-E", (1405, 455))
    draw_edge(parts, [(1610, 305), (1810, 345)], "分支F：退出请求", (1775, 285))
    draw_edge(parts, [(760, 605), (500, 565)], "调用", (625, 560))
    draw_edge(parts, [(760, 635), (640, 725)], "查询", (700, 705))
    draw_edge(parts, [(1560, 605), (1740, 565)], "查询", (1660, 555))
    draw_edge(parts, [(1560, 635), (1740, 725)], "调用", (1660, 705))
    draw_edge(parts, [(1160, 650), (1160, 880)], "输出", (1220, 780))

    write_svg_and_png("analysis_flow", 2500, 1100, "EnzyWizard-Agent 分析模块", parts)


def main() -> None:
    render_overall_flow()
    render_planning_flow()
    render_analysis_flow()


if __name__ == "__main__":
    main()

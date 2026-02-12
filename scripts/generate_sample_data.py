"""サンプルデータ生成スクリプト"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent


def generate_scratch(draw, width, height):
    """擦り傷（線状）を生成"""
    x1 = random.randint(50, width - 100)
    y1 = random.randint(50, height - 100)
    length = random.randint(80, 200)
    angle = random.uniform(0, 2 * np.pi)
    x2 = x1 + int(length * np.cos(angle))
    y2 = y1 + int(length * np.sin(angle))

    line_width = random.randint(2, 6)
    color = random.randint(60, 120)
    draw.line([(x1, y1), (x2, y2)], fill=(color, color, color), width=line_width)

    return "擦り傷", "線状"


def generate_dent(draw, width, height):
    """打痕（点状）を生成"""
    cx = random.randint(80, width - 80)
    cy = random.randint(80, height - 80)
    radius = random.randint(15, 40)

    for r in range(radius, 0, -2):
        shade = 100 + int((radius - r) * 3)
        shade = min(shade, 180)
        draw.ellipse(
            [(cx - r, cy - r), (cx + r, cy + r)],
            fill=(shade, shade, shade)
        )

    return "打痕", "点状"


def generate_corrosion(draw, width, height):
    """腐食（面状）を生成"""
    x = random.randint(50, width - 150)
    y = random.randint(50, height - 150)
    w = random.randint(60, 120)
    h = random.randint(60, 120)

    for _ in range(50):
        px = x + random.randint(0, w)
        py = y + random.randint(0, h)
        size = random.randint(3, 10)
        color = random.randint(80, 140)
        draw.ellipse(
            [(px, py), (px + size, py + size)],
            fill=(color, color - 10, color - 20)
        )

    return "腐食", "面状"


def generate_cut(draw, width, height):
    """切り傷（線状）を生成"""
    x1 = random.randint(50, width - 100)
    y1 = random.randint(50, height - 100)
    length = random.randint(50, 150)
    angle = random.uniform(0, 2 * np.pi)
    x2 = x1 + int(length * np.cos(angle))
    y2 = y1 + int(length * np.sin(angle))

    draw.line([(x1, y1), (x2, y2)], fill=(40, 40, 40), width=1)
    draw.line([(x1 + 1, y1), (x2 + 1, y2)], fill=(80, 80, 80), width=1)

    return "切り傷", "線状"


def generate_burn(draw, width, height):
    """焼け（面状）を生成"""
    cx = random.randint(80, width - 80)
    cy = random.randint(80, height - 80)

    for _ in range(30):
        px = cx + random.randint(-50, 50)
        py = cy + random.randint(-50, 50)
        size = random.randint(5, 20)
        r = random.randint(60, 100)
        g = random.randint(40, 70)
        b = random.randint(30, 50)
        draw.ellipse([(px, py), (px + size, py + size)], fill=(r, g, b))

    return "焼け", "面状"


def generate_base_image(width=224, height=224):
    """ベースとなる金属表面風の画像を生成"""
    base_color = random.randint(160, 200)
    img = Image.new("RGB", (width, height), (base_color, base_color, base_color))

    pixels = np.array(img)
    noise = np.random.normal(0, 8, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)

    return img


def generate_sample_image(defect_type=None):
    """サンプル画像を生成"""
    img = generate_base_image()
    draw = ImageDraw.Draw(img)

    generators = [
        generate_scratch,
        generate_dent,
        generate_corrosion,
        generate_cut,
        generate_burn,
    ]

    if defect_type is None:
        generator = random.choice(generators)
    else:
        generator = generators[defect_type]

    cause, shape = generator(draw, img.width, img.height)

    # 深さをランダムに決定
    depths = ["表層", "中層", "深層"]
    depth = random.choice(depths)

    # 少しぼかしを追加
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img, cause, shape, depth


def main():
    """サンプルデータを生成"""
    # 出力ディレクトリ
    train_dir = PROJECT_ROOT / "data" / "processed" / "train"
    val_dir = PROJECT_ROOT / "data" / "processed" / "val"

    train_images_dir = train_dir / "images"
    val_images_dir = val_dir / "images"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)

    print("サンプルデータを生成中...")

    # 学習データ生成
    train_samples = []
    for i in range(100):
        img, cause, shape, depth = generate_sample_image()
        filename = f"{i:05d}.jpg"
        img.save(train_images_dir / filename, quality=95)

        train_samples.append({
            "id": f"{i:05d}",
            "image_path": f"images/{filename}",
            "cause": cause,
            "shape": shape,
            "depth": depth,
        })

        if (i + 1) % 20 == 0:
            print(f"  学習データ: {i + 1}/100")

    # 学習データのアノテーション保存
    train_annotation = {
        "version": "1.0",
        "samples": train_samples,
    }
    with open(train_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(train_annotation, f, ensure_ascii=False, indent=2)

    # 検証データ生成
    val_samples = []
    for i in range(20):
        img, cause, shape, depth = generate_sample_image()
        filename = f"{i:05d}.jpg"
        img.save(val_images_dir / filename, quality=95)

        val_samples.append({
            "id": f"{i:05d}",
            "image_path": f"images/{filename}",
            "cause": cause,
            "shape": shape,
            "depth": depth,
        })

    print(f"  検証データ: 20/20")

    # 検証データのアノテーション保存
    val_annotation = {
        "version": "1.0",
        "samples": val_samples,
    }
    with open(val_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(val_annotation, f, ensure_ascii=False, indent=2)

    print("\n完了!")
    print(f"  学習データ: {len(train_samples)} 枚 -> {train_dir}")
    print(f"  検証データ: {len(val_samples)} 枚 -> {val_dir}")

    # 統計情報
    print("\n統計情報:")
    for name, samples in [("学習", train_samples), ("検証", val_samples)]:
        cause_counts = {}
        shape_counts = {}
        depth_counts = {}
        for s in samples:
            cause_counts[s["cause"]] = cause_counts.get(s["cause"], 0) + 1
            shape_counts[s["shape"]] = shape_counts.get(s["shape"], 0) + 1
            depth_counts[s["depth"]] = depth_counts.get(s["depth"], 0) + 1

        print(f"\n  {name}データ:")
        print(f"    原因: {cause_counts}")
        print(f"    形状: {shape_counts}")
        print(f"    深さ: {depth_counts}")


if __name__ == "__main__":
    main()

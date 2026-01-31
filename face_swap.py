import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
import insightface
import argparse


def face_swap(source_img_path, target_img_path, output_path="output_swap.jpg"):
    # Load face detection + recognition model
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load inswapper model
    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx",
        download=False,
        download_zip=False
    )

    # Load images
    img_source = cv2.imread(source_img_path)
    img_target = cv2.imread(target_img_path)

    if img_source is None or img_target is None:
        print("Không thể đọc ảnh. Kiểm tra đường dẫn!")
        return None

    # Detect faces
    source_faces = app.get(img_source)
    target_faces = app.get(img_target)

    if len(source_faces) == 0:
        print("Không tìm thấy mặt trong ảnh nguồn!")
        return None
    if len(target_faces) == 0:
        print("Không tìm thấy mặt trong ảnh đích!")
        return None

    source_face  = source_faces[0]  # Lấy khuôn mặt đích đầu tiên
    result = img_target.copy()

    # Thay mặt lần lượt cho từng khuôn mặt trong ảnh nguồn
    for face in target_faces:
        result = swapper.get(result, face, source_face, paste_back=True)

    # Lưu kết quả
    cv2.imwrite(output_path, result)

    print("Hoàn thành! Ảnh đã lưu tại:", output_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Swap CLI")
    parser.add_argument("--source", type=str, required=True,
                        help="Đường dẫn ảnh nguồn (ảnh cung cấp khuôn mặt)")
    parser.add_argument("--target", type=str, required=True,
                        help="Đường dẫn ảnh đích (ảnh cần thay mặt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Đường dẫn file xuất ảnh")
    args = parser.parse_args()
    face_swap(args.source, args.target, args.output)

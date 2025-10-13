from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.io import decode_png, read_file
import tensorflow as tf
from tensorflow.image import resize
from matplotlib.pyplot import figure, title, axis, imshow, show


def load_and_preprocess_image(file_path):
    """이미지 파일을 읽고 전처리합니다."""
    image = decode_png(read_file(file_path), channels=1)  # PNG 파일을 흑백 이미지로 디코딩
    # 자르기: 세로 방향은 230~930행, 가로 방향은 530~1370열 선택
    cropped = image[230:930, 530:1370]
    # 특정 열 범위 선택
    selected_columns = tf.concat((
        cropped[:, :40],  # 원본 이미지 기준: 530~570 열
        cropped[:, 390:420],  # 920~950 열
        cropped[:, 450:480],  # 980~1010 열
        cropped[:, 500:530],  # 1030~1060 열
        cropped[:, 570:640],  # 1100~1170 열
        cropped[:, 670:740],  # 1200~1270 열
        cropped[:, 770:]  # 1300 열 이후부터 끝까지
    ), axis=1)
    # 크기를 (400, 400)으로 조정 (종횡비 유지)
    resized = resize(selected_columns, (400, 400), preserve_aspect_ratio=True)
    return resized, file_path


def display_image(image_tensor, path_tensor):
    """단일 흑백 이미지 텐서를 화면에 표시합니다."""
    figure(figsize=(6, 6))
    # 텐서를 문자열로 디코딩
    path_str = path_tensor.numpy().decode('utf-8')
    title(path_str)  # 제목에 파일 경로 표시
    axis('off')  # 축 숨기기
    imshow(image_tensor, cmap='gray')  # 이미지 표시 (회색조)
    show()


# 'datasets' 폴더 내의 모든 하위 디렉토리 및 파일 목록 생성
dataset = Dataset.list_files('datasets/*/*')
# 이미지 로드 및 전처리 함수 적용 (병렬 처리)
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# 배치 크기 32로 묶고, 성능을 위한 prefetch 설정
dataset = dataset.batch(32).prefetch(AUTOTUNE)
# 첫 번째 배치에서 첫 번째 이미지를 표시
for images, paths in dataset.take(1):
    display_image(images[0], paths[0])

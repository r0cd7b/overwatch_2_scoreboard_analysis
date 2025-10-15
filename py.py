from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.io import decode_png, read_file
import tensorflow as tf
from tensorflow.image import resize
from tensorflow.strings import split, upper, substr, to_number
from tensorflow import int32
from matplotlib.pyplot import figure, title, axis, imshow, show


def load_and_process_image(file_path):
    # PNG 파일을 로드하고 디코딩 (그레이스케일)
    image = decode_png(read_file(file_path), channels=1)

    # 관심 영역(Region of Interest) 자르기
    cropped = image[230:930, 530:1370]

    # 특정 열 부분을 추출하고 이어 붙이기
    columns = (
        cropped[:, :40],  # 왼쪽 여백
        cropped[:, 390:420],  # 문자 1
        cropped[:, 450:480],  # 문자 2
        cropped[:, 500:530],  # 문자 3
        cropped[:, 570:640],  # 문자 4
        cropped[:, 670:740],  # 문자 5
        cropped[:, 770:]  # 오른쪽 여백
    )
    combined = tf.concat(columns, axis=1)  # 가로 방향으로 이어 붙이기

    # 크기 조정 (400x400), 비율 유지
    resized = resize(combined, size=(400, 400), preserve_aspect_ratio=True)

    # 파일 이름에서 메타데이터 추출
    filename_parts = split(file_path, '/')[-1]  # 경로에서 파일 이름만 추출
    parts = split(filename_parts, '_')
    label_str = upper(substr(parts[0], 0, 2))  # 첫 두 글자를 대문자로 변환 (예: 라벨 코드)
    number = to_number(split(parts[1], '.')[0], out_type=int32)  # 숫자 부분 추출 및 정수형 변환

    return resized, label_str, number


def visualize_batch(images, labels, numbers):
    figure(figsize=(6, 6))
    title(f'{labels[0].numpy().decode()} {numbers[0]}')  # 첫 번째 이미지의 라벨과 숫자 표시
    axis('off')
    imshow(images[0], cmap='gray')  # 첫 번째 이미지 표시
    show()


# 데이터셋 생성
dataset = Dataset.list_files('datasets/*/*')  # 모든 하위 폴더의 파일 불러오기
dataset = dataset.map(load_and_process_image, num_parallel_calls=AUTOTUNE)  # 병렬 처리로 이미지 전처리
dataset = dataset.batch(32).prefetch(AUTOTUNE)  # 배치 구성 및 사전 로딩

# 배치 하나 시각화
for images, labels, numbers in dataset.take(1):
    visualize_batch(images, labels, numbers)

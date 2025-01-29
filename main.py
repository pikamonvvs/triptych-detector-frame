import os
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2


def process_video(video_path, output_dir, frame_interval_seconds, divisor):
    """단일 비디오 처리를 위한 헬퍼 함수"""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"에러: 비디오 파일을 열 수 없습니다 {video_path}")
        return 0

    # 비디오 읽기 버퍼 크기 증가
    video.set(cv2.CAP_PROP_BUFFERSIZE, 4096)

    filename = os.path.basename(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"에러: {video_path}의 프레임 레이트를 가져올 수 없습니다.")
        return 0

    frame_interval = int(video_fps * frame_interval_seconds)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_frame_count = 0

    print(f"[{filename}] 처리 시작 - 총 {total_frames}프레임")

    # PNG 압축 최적화 파라미터
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 0-9 사이값, 낮을수록 빠르고 파일 크기는 커짐

    # 프레임 간격만큼 건너뛰면서 처리
    for frame_count in range(0, total_frames, frame_interval):
        # 원하는 프레임 위치로 직접 이동
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = video.read()

        if not ret:
            break

        # 프레임 크기 조절
        height, width = frame.shape[:2]
        new_height = height // divisor
        new_width = width // divisor

        # SIMD 최적화된 리사이징 사용
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        frame_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_frame_{saved_frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame, png_params)
        saved_frame_count += 1

    video.release()
    print(f"[{filename}] 처리 완료. 저장된 프레임: {saved_frame_count}")
    return saved_frame_count


def extract_frames(video_dir_paths, output_dir, frame_interval_seconds, divisor):
    """
    여러 폴더에서 동영상 파일의 프레임을 추출하여 이미지 파일로 저장합니다.

    Args:
        video_dir_paths (list): 동영상 파일이 있는 폴더 경로 목록
        output_dir (str): 추출된 프레임이 저장될 디렉토리
        frame_interval_seconds (int): 프레임을 추출할 시간 간격(초)
        divisor (int): 프레임 크기를 나눌 값

    Returns:
        None
    """
    print("프레임 추출 작업을 시작합니다...")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 모든 비디오 파일 경로 수집
    video_paths = []
    for video_dir in video_dir_paths:
        for filename in os.listdir(video_dir):
            if filename.lower().endswith((".mp4", ".ts")):
                video_paths.append(os.path.join(video_dir, filename))

    total_videos = len(video_paths)
    print(f"총 {total_videos}개의 비디오 파일이 발견되었습니다.")

    # 멀티프로세싱 설정
    num_processes = max(cpu_count() - 1, 1)
    print(f"CPU 코어 {num_processes}개를 사용하여 병렬 처리를 시작합니다.")

    process_video_partial = partial(process_video, output_dir=output_dir, frame_interval_seconds=frame_interval_seconds, divisor=divisor)

    # 병렬 처리 실행
    with Pool(num_processes) as pool:
        results = pool.map(process_video_partial, video_paths)

    total_saved_frames = sum(results)

    # 생성된 프레임 파일들의 총 용량 계산
    total_size = 0
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            file_path = os.path.join(output_dir, filename)
            total_size += os.path.getsize(file_path)

    # 용량을 보기 좋게 변환 (B, KB, MB, GB)
    size_units = ["B", "KB", "MB", "GB"]
    size_index = 0
    formatted_size = total_size

    while formatted_size >= 1024 and size_index < len(size_units) - 1:
        formatted_size /= 1024
        size_index += 1

    print("모든 작업이 완료되었습니다!")
    print(f"처리된 비디오 파일: {total_videos}개")
    print(f"총 저장된 프레임: {total_saved_frames}개")
    print(f"생성된 프레임 파일의 총 용량: {formatted_size:.2f} {size_units[size_index]}")


# 사용 예시
if __name__ == "__main__":
    video_dirs = []
    while True:
        dir_path = input("동영상이 있는 폴더 경로를 입력하세요 (종료하려면 빈 값 입력): ").strip('"')
        if not dir_path:
            break
        if os.path.isdir(dir_path):
            video_dirs.append(dir_path)
            print(f"추가된 경로: {dir_path}")
        else:
            print("유효하지 않은 디렉토리 경로입니다.")

    if not video_dirs:
        print("최소 하나의 유효한 디렉토리를 입력해야 합니다.")
        exit()

    output_dir = input("프레임을 저장할 출력 디렉토리를 입력하세요: ").strip('"')
    frame_interval_seconds = int(input("프레임 추출 간격(초)을 입력하세요 (예: 1, 5, 10): "))
    divisor = int(input("프레임 크기를 나눌 값을 입력하세요 (예: 2는 크기를 1/2로 줄임): "))

    print("작업을 시작합니다...")
    start_time = time.time()
    extract_frames(video_dirs, output_dir, frame_interval_seconds, divisor)
    end_time = time.time()
    print(f"총 수행 시간: {end_time - start_time:.2f}초")

import os

import cv2


def extract_frames(video_path, output_dir, frame_interval_seconds):
    """
    Extracts frames from a video file at a specified time interval and saves them as image files.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        frame_interval_seconds (int): Time interval in seconds between frames to extract.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    video_fps = video.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print("Error: Cannot retrieve frame rate of the video.")
        return

    frame_interval = int(video_fps * frame_interval_seconds)  # Calculate frame interval

    # Read and save frames
    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            print(f"Saved: {frame_filename}")

        frame_count += 1

    # Release resources
    video.release()
    print(f"Extraction complete. Total frames saved: {saved_frame_count}")


# Example usage
if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip('"')
    output_dir = input("Enter the output directory for the frames: ").strip('"')
    frame_interval_seconds = int(input("Enter the frame interval in seconds (e.g., 1, 5, 10): "))

    extract_frames(video_path, output_dir, frame_interval_seconds)

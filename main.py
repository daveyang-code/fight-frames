import cv2
import mediapipe as mp
import numpy as np
import time
import argparse


class FrameAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Constants for punch detection
        self.MOVEMENT_THRESHOLD = 0.05  # Threshold for detecting significant movement
        self.DIRECTION_CHANGE_THRESHOLD = (
            0.03  # Threshold for detecting direction change
        )

        # Punch state variables
        self.punch_state = "ready"  # ready, extension, retraction
        self.extension_frames = 0
        self.retraction_frames = 0
        self.fps = 0

        # Hand position tracking
        self.prev_right_hand = None
        self.prev_left_hand = None
        self.initial_right_hand = None  # Starting position for right hand
        self.initial_left_hand = None  # Starting position for left hand
        self.max_extension_pos = None  # Position at maximum extension

        # Velocity and direction tracking
        self.right_hand_velocity = 0
        self.left_hand_velocity = 0
        self.right_hand_direction = None  # "extending" or "retracting"
        self.left_hand_direction = None  # "extending" or "retracting"
        self.prev_right_direction = None
        self.prev_left_direction = None

        # Frame counting and timing
        self.frame_count = 0
        self.start_time = time.time()

        # History for smoothing
        self.right_velocity_history = []
        self.left_velocity_history = []
        self.history_size = 3

        # Punch history for stats
        self.punch_history = []
        self.max_history = 10

        # Active hand (which hand is punching)
        self.active_hand = None

        # Distance tracking for extension/retraction detection
        self.start_position = None
        self.extended_position = None
        self.current_distance = 0
        self.max_distance = 0
        self.distance_history = []

    def calculate_velocity(self, current_pos, previous_pos):
        """Calculate the velocity between two positions"""
        if previous_pos is None:
            return 0
        return np.linalg.norm(np.array(current_pos) - np.array(previous_pos))

    def calculate_direction(self, current_pos, previous_pos, initial_pos):
        """Calculate if the hand is extending or retracting"""
        if previous_pos is None or initial_pos is None:
            return None

        # Calculate distance from initial position
        current_distance = np.linalg.norm(np.array(current_pos) - np.array(initial_pos))
        previous_distance = np.linalg.norm(
            np.array(previous_pos) - np.array(initial_pos)
        )

        # If current distance is greater than previous, hand is extending
        if current_distance > previous_distance:
            return "extending"
        else:
            return "retracting"

    def smooth_velocity(self, velocity, hand_side):
        """Apply smoothing to velocity calculations"""
        if hand_side == "right":
            self.right_velocity_history.append(velocity)
            if len(self.right_velocity_history) > self.history_size:
                self.right_velocity_history.pop(0)
            return sum(self.right_velocity_history) / len(self.right_velocity_history)
        else:
            self.left_velocity_history.append(velocity)
            if len(self.left_velocity_history) > self.history_size:
                self.left_velocity_history.pop(0)
            return sum(self.left_velocity_history) / len(self.left_velocity_history)

    def analyze_frame(self, frame, frame_count):
        """Analyze a single frame for punch detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Calculate FPS
        self.frame_count += 1
        if self.frame_count >= 10:
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        h, w, _ = frame.shape
        right_hand_pos = None
        left_hand_pos = None

        # Extract hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Determine if this is a right or left hand
                handedness = results.multi_handedness[hand_idx].classification[0].label

                # Get middle knuckle position as reference point for the hand
                middle_knuckle = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                ]
                hand_pos = (middle_knuckle.x * w, middle_knuckle.y * h)

                if handedness == "Right":
                    right_hand_pos = hand_pos
                    # Initialize start position if not set
                    if self.initial_right_hand is None:
                        self.initial_right_hand = hand_pos
                else:
                    left_hand_pos = hand_pos
                    # Initialize start position if not set
                    if self.initial_left_hand is None:
                        self.initial_left_hand = hand_pos

        # Process right hand movement
        if right_hand_pos is not None:
            # Calculate velocity
            self.right_hand_velocity = self.calculate_velocity(
                right_hand_pos, self.prev_right_hand
            )
            self.right_hand_velocity = self.smooth_velocity(
                self.right_hand_velocity, "right"
            )

            # Calculate direction
            if self.prev_right_hand is not None and self.initial_right_hand is not None:
                self.prev_right_direction = self.right_hand_direction
                self.right_hand_direction = self.calculate_direction(
                    right_hand_pos, self.prev_right_hand, self.initial_right_hand
                )

            self.prev_right_hand = right_hand_pos

        # Process left hand movement
        if left_hand_pos is not None:
            # Calculate velocity
            self.left_hand_velocity = self.calculate_velocity(
                left_hand_pos, self.prev_left_hand
            )
            self.left_hand_velocity = self.smooth_velocity(
                self.left_hand_velocity, "left"
            )

            # Calculate direction
            if self.prev_left_hand is not None and self.initial_left_hand is not None:
                self.prev_left_direction = self.left_hand_direction
                self.left_hand_direction = self.calculate_direction(
                    left_hand_pos, self.prev_left_hand, self.initial_left_hand
                )

            self.prev_left_hand = left_hand_pos

        # Determine active hand based on velocity
        if right_hand_pos is not None and left_hand_pos is not None:
            if self.right_hand_velocity > self.left_hand_velocity:
                self.active_hand = "right"
                active_velocity = self.right_hand_velocity
                active_direction = self.right_hand_direction
                prev_active_direction = self.prev_right_direction
                active_pos = right_hand_pos
                initial_pos = self.initial_right_hand
            else:
                self.active_hand = "left"
                active_velocity = self.left_hand_velocity
                active_direction = self.left_hand_direction
                prev_active_direction = self.prev_left_direction
                active_pos = left_hand_pos
                initial_pos = self.initial_left_hand
        elif right_hand_pos is not None:
            self.active_hand = "right"
            active_velocity = self.right_hand_velocity
            active_direction = self.right_hand_direction
            prev_active_direction = self.prev_right_direction
            active_pos = right_hand_pos
            initial_pos = self.initial_right_hand
        elif left_hand_pos is not None:
            self.active_hand = "left"
            active_velocity = self.left_hand_velocity
            active_direction = self.left_hand_direction
            prev_active_direction = self.prev_left_direction
            active_pos = left_hand_pos
            initial_pos = self.initial_left_hand
        else:
            # No hands detected
            return frame

        # Calculate distance from starting position
        if active_pos is not None and initial_pos is not None:
            current_distance = np.linalg.norm(
                np.array(active_pos) - np.array(initial_pos)
            )
            self.current_distance = current_distance

            # Track maximum distance (full extension)
            if self.max_distance < current_distance:
                self.max_distance = current_distance
                self.extended_position = active_pos

            # Add to distance history for visualization
            self.distance_history.append(current_distance)
            if len(self.distance_history) > 100:  # Keep last 100 frames
                self.distance_history.pop(0)

        # State machine for punch tracking
        if self.punch_state == "ready":
            # Detect start of extension (significant velocity + extending direction)
            if (
                active_velocity > self.MOVEMENT_THRESHOLD
                and active_direction == "extending"
            ):
                self.punch_state = "extension"
                self.extension_frames = 1
                self.max_distance = 0  # Reset max distance

        elif self.punch_state == "extension":
            self.extension_frames += 1

            # Detect change to retraction (direction change or reaching peak extension)
            direction_changed = (
                active_direction == "retracting"
                and prev_active_direction == "extending"
            )
            velocity_dropped = active_velocity < self.DIRECTION_CHANGE_THRESHOLD

            if direction_changed or velocity_dropped:
                self.punch_state = "retraction"
                self.retraction_frames = 1

        elif self.punch_state == "retraction":
            self.retraction_frames += 1

            # Detect return to ready state (velocity low and close to starting position)
            close_to_start = (
                self.current_distance < self.max_distance * 0.2
            )  # Within 20% of max distance
            velocity_low = active_velocity < self.MOVEMENT_THRESHOLD

            if close_to_start or velocity_low:
                # Record punch data
                punch_data = {
                    "extension_frames": self.extension_frames,
                    "retraction_frames": self.retraction_frames,
                    "total_frames": self.extension_frames + self.retraction_frames,
                    "hand": self.active_hand,
                    "max_distance": self.max_distance,
                }
                self.punch_history.append(punch_data)
                if len(self.punch_history) > self.max_history:
                    self.punch_history.pop(0)

                # Reset state
                self.punch_state = "ready"
                self.extension_frames = 0
                self.retraction_frames = 0

        # Draw information on frame
        self.draw_info(frame)

        # Draw distance graph
        self.draw_distance_graph(frame)

        return frame

    def draw_distance_graph(self, frame):
        """Draw a graph showing hand distance from starting position"""
        if not self.distance_history:
            return

        h, w, _ = frame.shape
        graph_width = 200
        graph_height = 100
        graph_x = w - graph_width - 10
        graph_y = 10

        # Draw graph background
        cv2.rectangle(
            frame,
            (graph_x, graph_y),
            (graph_x + graph_width, graph_y + graph_height),
            (0, 0, 0),
            -1,
        )

        # Draw axes
        cv2.line(
            frame,
            (graph_x, graph_y + graph_height),
            (graph_x + graph_width, graph_y + graph_height),
            (255, 255, 255),
            1,
        )
        cv2.line(
            frame,
            (graph_x, graph_y),
            (graph_x, graph_y + graph_height),
            (255, 255, 255),
            1,
        )

        # Draw graph title
        cv2.putText(
            frame,
            "Hand Distance",
            (graph_x, graph_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw graph values
        max_value = max(self.distance_history) if self.distance_history else 1
        points = []

        for i, value in enumerate(self.distance_history[-graph_width:]):
            x = graph_x + i * (
                graph_width / min(len(self.distance_history), graph_width)
            )
            y = graph_y + graph_height - (value / max_value) * graph_height
            if not np.isnan(y):  # Skip if y is NaN
                points.append((int(x), int(y)))

        # Draw line graph
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 1)

    def draw_info(self, frame):
        """Draw punch information on the frame"""
        h, w, _ = frame.shape

        # Display velocities
        if self.active_hand == "right":
            velocity_color = (
                (0, 255, 0)
                if self.right_hand_velocity > self.MOVEMENT_THRESHOLD
                else (255, 255, 255)
            )
            cv2.putText(
                frame,
                f"Right Hand Velocity: {self.right_hand_velocity:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                velocity_color,
                2,
            )
            cv2.putText(
                frame,
                f"Left Hand Velocity: {self.left_hand_velocity:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        else:
            velocity_color = (
                (0, 255, 0)
                if self.left_hand_velocity > self.MOVEMENT_THRESHOLD
                else (255, 255, 255)
            )
            cv2.putText(
                frame,
                f"Right Hand Velocity: {self.right_hand_velocity:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Left Hand Velocity: {self.left_hand_velocity:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                velocity_color,
                2,
            )

        # Display active hand and direction
        active_direction = (
            self.right_hand_direction
            if self.active_hand == "right"
            else self.left_hand_direction
        )
        direction_str = active_direction if active_direction else "neutral"
        cv2.putText(
            frame,
            f"Active Hand: {self.active_hand.capitalize() if self.active_hand else 'None'}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Direction: {direction_str.capitalize() if direction_str else 'Unknown'}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display punch state with different colors
        if self.punch_state == "ready":
            status_color = (0, 255, 0)  # Green
            status_text = "Ready"
        elif self.punch_state == "extension":
            status_color = (0, 0, 255)  # Red
            status_text = f"Extension: {self.extension_frames} frames"
        else:  # retraction
            status_color = (255, 0, 0)  # Blue
            status_text = f"Retraction: {self.retraction_frames} frames"

        cv2.putText(
            frame,
            f"State: {status_text}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        # Display distance data
        cv2.putText(
            frame,
            f"Current Distance: {self.current_distance:.2f}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Max Distance: {self.max_distance:.2f}",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display punch history statistics
        if self.punch_history:
            avg_extension = sum(
                p["extension_frames"] for p in self.punch_history
            ) / len(self.punch_history)
            avg_retraction = sum(
                p["retraction_frames"] for p in self.punch_history
            ) / len(self.punch_history)

            cv2.putText(
                frame,
                f"Avg Extension: {avg_extension:.1f} frames",
                (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Avg Retraction: {avg_retraction:.1f} frames",
                (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Total Punches: {len(self.punch_history)}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def analyze_video(self, video_source=0):
        """Analyze video from file or webcam"""
        cap = cv2.VideoCapture(video_source)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            processed_frame = self.analyze_frame(frame, frame_count)

            cv2.imshow("Punch Analyzer", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

        # Print final stats
        print("\n===== Punch Analysis Results =====")
        if self.punch_history:
            avg_extension = sum(
                p["extension_frames"] for p in self.punch_history
            ) / len(self.punch_history)
            avg_retraction = sum(
                p["retraction_frames"] for p in self.punch_history
            ) / len(self.punch_history)
            avg_total = sum(p["total_frames"] for p in self.punch_history) / len(
                self.punch_history
            )

            right_punches = sum(1 for p in self.punch_history if p["hand"] == "right")
            left_punches = sum(1 for p in self.punch_history if p["hand"] == "left")

            print(f"Average Extension Frames: {avg_extension:.1f}")
            print(f"Average Retraction Frames: {avg_retraction:.1f}")
            print(f"Average Total Frames: {avg_total:.1f}")
            print(f"Number of Punches Analyzed: {len(self.punch_history)}")
            print(f"Right Hand Punches: {right_punches}")
            print(f"Left Hand Punches: {left_punches}")
        else:
            print("No punches detected in this session.")


def main():
    parser = argparse.ArgumentParser(
        description="Punch Extension & Retraction Analyzer"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (0 for webcam, or path to video file)",
    )
    args = parser.parse_args()

    # Convert '0' to integer 0 for webcam
    video_source = 0 if args.source == "0" else args.source

    analyzer = FrameAnalyzer()
    analyzer.analyze_video(video_source)


if __name__ == "__main__":
    main()

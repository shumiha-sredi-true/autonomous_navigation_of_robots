from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        # Сначала стартуем камеру
        Node(
            package='ros2_apriltag',
            executable='camera_node',
            name='camera_node'
        ),
        # Немного задерживаем старт Apriltag, чтобы камера успела запуститься
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='ros2_apriltag',
                    executable='apriltag_node',
                    name='apriltag_node'
                )
            ]
        ),
        # WS нода может стартовать параллельно
        Node(
            package='ros2_apriltag',
            executable='ws_node',
            name='ws_node'
        ),
        # LiveKit нода также может стартовать параллельно
        Node(
            package='ros2_apriltag',
            executable='livekit_node',
            name='livekit_node'
        ),
        # Script execution нода должна стартовать после камеры
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='ros2_apriltag',
                    executable='script_exec_node',
                    name='script_exec_node'
                )
            ]
        )
    ])

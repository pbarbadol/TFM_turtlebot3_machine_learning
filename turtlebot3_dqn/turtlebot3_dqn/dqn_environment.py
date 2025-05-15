#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal


class RLEnvironment(Node):

    def __init__(self):
        super().__init__('rl_environment')
        self.train_mode = True
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        self.max_step = 800

        self.done = False
        self.fail = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.scan_ranges = []
        self.min_obstacle_distance = 10.0
        self.min_obstacle_angle = 0.0     # NUEVO
        self.robot_roll = 0.0             # NUEVO
        self.robot_pitch = 0.0            # NUEVO
        self.max_roll_pitch = 0.8         # NUEVO: umbral de 45 grados para vuelco

        self.local_step = 0
        self.stop_cmd_vel_timer = None
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]

        qos = QoSProfile(depth=10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_sub_callback,
            qos_profile_sensor_data
        )

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            'task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            'task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            'initialize_env',
            callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Dqn,
            'rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            'make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn,
            'reset_environment',
            self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'service for initialize the environment is not available, waiting ...'
            )
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal.success:
            self.get_logger().error('initialize environment request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(
                'goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y)
            )

        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        response.state = state

        return response

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task succeed finished')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task failed finished')
        else:
            self.get_logger().error('task failed service call failed')

    def scan_sub_callback(self, scan):
        # Temporalmente almacena los 360 rangos procesados
        processed_scan_ranges_full = []
        num_of_lidar_rays = len(scan.ranges) # Debería ser 360 para TurtleBot3

        for i in range(num_of_lidar_rays):
            if scan.ranges[i] == float('Inf'):
                processed_scan_ranges_full.append(3.5)
            elif numpy.isnan(scan.ranges[i]):
                processed_scan_ranges_full.append(0.0)
            else:
                processed_scan_ranges_full.append(scan.ranges[i])

        # Reducir a 24 muestras (self.scan_ranges ahora tendrá 24 elementos)
        self.scan_ranges = [] # Asegúrate de que esté vacío antes de llenarlo
        if num_of_lidar_rays == 360:
            samples_per_group = 15 # 360 / 24 = 15
            for i in range(0, num_of_lidar_rays, samples_per_group):
                self.scan_ranges.append(min(processed_scan_ranges_full[i:i + samples_per_group]))
        else:
            self.get_logger().warn(
                f"Expected 360 laser scan ranges, but got {num_of_lidar_rays}. Filling scan_ranges with 24 zeros."
            )
            self.scan_ranges = [0.0] * 24

        # Calcular min_obstacle_distance y min_obstacle_angle a partir de las 24 muestras
        if self.scan_ranges:
            self.min_obstacle_distance = min(self.scan_ranges)
            min_obstacle_index = numpy.argmin(self.scan_ranges)
            # Cálculo del ángulo relativo al robot. Ajusta según sea necesario para tu convención.
            self.min_obstacle_angle = (min_obstacle_index * (2 * math.pi / 24.0)) - math.pi
        else:
            self.min_obstacle_distance = 3.5
            self.min_obstacle_angle = 0.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        #_, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_roll, self.robot_pitch, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        state = []
        # Asegurarse de que self.scan_ranges tenga 24 elementos (de scan_sub_callback)
        for scan_val in self.scan_ranges:
            state.append(float(scan_val))

        # Información del objetivo
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))

        # Información del obstáculo más cercano
        state.append(float(self.min_obstacle_distance))
        state.append(float(self.min_obstacle_angle))

        # DEBUG: Puedes descomentar esto para verificar el tamaño del estado antes de devolverlo
        # self.get_logger().info(f"calculate_state: len(self.scan_ranges)={len(self.scan_ranges)}, len(state)={len(state)}")

        self.local_step += 1

        # --- Condiciones de Fin de Episodio ---
        # 1. Meta Alcanzada
        if self.goal_distance < 0.20:
            if not self.done: # Solo si no está ya 'done'
                self.get_logger().info('Goal Reached')
                self.succeed = True
                self.done = True
                self.cmd_vel_pub.publish(Twist())
                self.local_step = 0
                self.call_task_succeed()

        # 2. Colisión (por LIDAR)
        # Ajusta este umbral si quieres que sea más sensible
        if self.min_obstacle_distance < 0.18: # Probando con 0.18 (antes 0.15)
            if not self.done: # Solo si no está ya 'done'
                self.get_logger().info(f'Collision (LIDAR) happened! Min_obstacle_distance: {self.min_obstacle_distance:.2f}')
                self.fail = True
                self.done = True
                self.cmd_vel_pub.publish(Twist())
                self.local_step = 0
                self.call_task_failed()

        # 3. Vuelco (por Roll/Pitch)
        if abs(self.robot_roll) > self.max_roll_pitch or abs(self.robot_pitch) > self.max_roll_pitch:
            if not self.done: # Solo si no está ya 'done'
                self.get_logger().info(f"Robot TIPPED OVER! Roll: {self.robot_roll:.2f}, Pitch: {self.robot_pitch:.2f}")
                self.fail = True
                self.done = True
                self.cmd_vel_pub.publish(Twist())
                self.local_step = 0
                self.call_task_failed()

        # 4. Timeout (Máximo de pasos)
        if self.local_step == self.max_step:
            if not self.done: # Solo si no está ya 'done'
                self.get_logger().info('Time out!')
                self.fail = True
                self.done = True
                self.cmd_vel_pub.publish(Twist())
                self.local_step = 0
                self.call_task_failed()

        return state

    def calculate_reward(self):
        if self.train_mode:

            if not hasattr(self, 'prev_goal_distance'):
                self.prev_goal_distance = self.init_goal_distance

            distance_reward = self.prev_goal_distance - self.goal_distance
            self.prev_goal_distance = self.goal_distance

            yaw_reward = (1 - 2 * math.sqrt(math.fabs(self.goal_angle / math.pi)))

            obstacle_reward = 0.0
            if self.min_obstacle_distance < 0.50:
                obstacle_reward = -1.0

            reward = (distance_reward * 10) + (yaw_reward / 5) + obstacle_reward

            if self.succeed:
                reward = 30.0
            elif self.fail:
                reward = -10.0

        else:
            if self.succeed:
                reward = 5.0
            elif self.fail:
                reward = -5.0
            else:
                reward = 0.0

        return reward

    def rl_agent_interface_callback(self, request, response):
        action = request.action
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = self.angular_vel[action]
        self.cmd_vel_pub.publish(twist)
        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(1.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(1.8, self.timer_callback)

        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        response.done = self.done

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        self.get_logger().info('Stop called')
        self.cmd_vel_pub.publish(Twist())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    rl_environment = RLEnvironment()
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Cross-ROS_DOMAIN_ID opponent-position bridge (2-car).

Both cars localize on the SAME map and publish their pose on /car_state/odom,
each in its OWN ROS_DOMAIN_ID (so their internal DDS traffic stays isolated).
This node pulls the OTHER car's /car_state/odom from the opponent's domain and
republishes it on THIS car's domain as /opponent_state/odom, throttled.

It runs two rclpy contexts in one process (the same trick domain_bridge uses):
  - ctx_opp : joined to the opponent's domain -> subscribes the opponent odom
  - ctx_mine: joined to our own domain        -> publishes /opponent_state/odom
A timer on our context republishes the latest received message at `rate` Hz
(throttle); it skips ticks with no fresh message (no duplicate spam) and stops
entirely if nothing has arrived within `timeout` s (so a dead/disconnected
opponent yields no stale ghost, letting the consumer notice it's gone).

Because both cars share the same map (same origin), the opponent's map-frame
pose is directly usable in our map frame -- no transform needed.

Parameters (on the OWN-domain node):
  opponent_domain (int)   : the other car's ROS_DOMAIN_ID. REQUIRED, must differ
                            from our own ROS_DOMAIN_ID.
  in_topic  (str)  = /car_state/odom       source topic on the opponent domain
  out_topic (str)  = /opponent_state/odom  republished topic on our domain
  rate (double)    = 50.0                  republish rate cap [Hz]
  timeout (double) = 0.5                   stop publishing if no msg within [s]

Our own domain is taken from the ROS_DOMAIN_ID env of the process.

Cross-machine discovery of the opponent domain relies on the same DDS transport
as everything else (CycloneDDS here): the opponent car's interface/IP must be
reachable/peered so ctx_opp can discover its publisher over the LAN.
"""
import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry


def main():
    my_domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))

    # ---- our-domain context: params, publisher, throttle timer ----
    ctx_mine = rclpy.Context()
    rclpy.init(context=ctx_mine, domain_id=my_domain)
    node = Node("opponent_bridge", context=ctx_mine)

    opponent_domain = int(node.declare_parameter("opponent_domain", -1).value)
    in_topic = str(node.declare_parameter("in_topic", "/car_state/odom").value)
    out_topic = str(node.declare_parameter("out_topic", "/opponent_state/odom").value)
    rate = float(node.declare_parameter("rate", 50.0).value)
    timeout = float(node.declare_parameter("timeout", 0.5).value)

    if opponent_domain < 0 or opponent_domain == my_domain:
        node.get_logger().error(
            f"opponent_domain must be set and differ from our domain "
            f"({my_domain}); got {opponent_domain}. Bridge disabled.")
        rclpy.shutdown(context=ctx_mine)
        return

    # best-effort: only the latest opponent pose matters; no need to retransmit
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    pub = node.create_publisher(Odometry, out_topic, qos)

    state = {"msg": None, "fresh": False, "t_rx": 0.0}
    lock = threading.Lock()

    def now_s():
        return node.get_clock().now().nanoseconds * 1e-9

    def on_timer():
        with lock:
            m = state["msg"]
            fresh = state["fresh"]
            t_rx = state["t_rx"]
            state["fresh"] = False
        if m is None or not fresh:
            return
        if (now_s() - t_rx) > timeout:  # opponent went silent -> stop republishing
            return
        pub.publish(m)

    node.create_timer(1.0 / max(1.0, rate), on_timer)

    # ---- opponent-domain context: subscriber ----
    ctx_opp = rclpy.Context()
    rclpy.init(context=ctx_opp, domain_id=opponent_domain)
    node_rx = Node("opponent_bridge_rx", context=ctx_opp)

    def on_odom(msg):
        with lock:
            state["msg"] = msg
            state["fresh"] = True
            state["t_rx"] = now_s()

    node_rx.create_subscription(Odometry, in_topic, on_odom, qos)

    node.get_logger().info(
        f"opponent bridge up: domain {opponent_domain} '{in_topic}' -> "
        f"domain {my_domain} '{out_topic}' @ {rate:.0f} Hz (timeout {timeout:.2f}s)")

    exec_opp = SingleThreadedExecutor(context=ctx_opp)
    exec_opp.add_node(node_rx)
    t = threading.Thread(target=exec_opp.spin, daemon=True)
    t.start()

    exec_mine = SingleThreadedExecutor(context=ctx_mine)
    exec_mine.add_node(node)
    try:
        exec_mine.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec_opp.shutdown()
        node_rx.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown(context=ctx_opp)
        rclpy.try_shutdown(context=ctx_mine)


if __name__ == "__main__":
    main()

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int16.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

using namespace std;

typedef nav_msgs::msg::Odometry Odometry;
typedef sensor_msgs::msg::LaserScan LaserScan;
typedef ackermann_msgs::msg::AckermannDriveStamped AckermannDriveStamped;

class OpponentPredictor : public rclcpp::Node {
// Implement opponent predictor on the car

public:
    OpponentPredictor() : Node("opponent_predictor_node") {
        // Subscribers
        rclcpp::Subscription<Odometry>::SharedPtr pose_sub_;
        rclcpp::Subscription<Odometry>::SharedPtr opp_pose_sub_;
        rclcpp::Subscription<LaserScan>::SharedPtr scan_sub_;

        // Publishers
        rclcpp::Publisher<AckermannDriveStamped>::SharedPtr drive_pub_;
        rclcpp::Publisher<AckermannDriveStamped>::SharedPtr opp_drive_pub_;
    }

private:
    string pose_topic = "/ego_racecar/odom";
    string scan_topic = "/scan";
    string drive_topic = "/drive";

    string opp_pose_topic = "/opp_racecar/odom";
    string opp_drive_topic = "/opp_drive";

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        (void) scan_msg;
    }

    void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
        (void) pose_msg;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpponentPredictor>());
    rclcpp::shutdown();
    return 0;
}
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ctime>
#include <cstdio>
#include <sstream>	
#include <iostream>

#define DEBUG
cv::Mat rgbMat, depthMat, xyzMat; 


void rgbCallback(const sensor_msgs::ImageConstPtr &msg_rgb) {

#ifdef DEBUG
	std::cerr << "call rgb " << '\n';
#endif
	cv_bridge::CvImagePtr rgbPtr = cv_bridge::toCvCopy(*msg_rgb, sensor_msgs::image_encodings::TYPE_8UC3);
    rgbMat = rgbPtr->image.clone();
    cv::cvtColor(rgbMat, rgbMat, cv::COLOR_BGR2RGB);
//	cv::imshow("rgb", rgbMat);
//	cv::waitKey(10);
#ifdef DEBUG
    double lastTime = ros::Time::now().toSec();
    std::cerr << "RGB : " << lastTime << '\n';
#endif    
}
void depthCallback(const sensor_msgs::ImageConstPtr &msg_depth) {
#ifdef DEBUG
	std::cerr << "call depth " << '\n';
#endif
	cv_bridge::CvImagePtr depthPtr = cv_bridge::toCvCopy(*msg_depth, sensor_msgs::image_encodings::TYPE_32FC1);
    depthMat = depthPtr->image.clone();
	// cv::imshow("depth", depthMat);
    // depth *= 5;
#ifdef DEBUG
    double lastTime = ros::Time::now().toSec();
    std::cerr << "DEPTH : " << lastTime << '\n';
#endif
}
int main(int argc, char **argv) {
    ros::init(argc, argv, "TestAstra");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
	image_transport::Subscriber rgb_sub, depth_sub;

    rgb_sub = it.subscribe("/camera/rgb/image_raw", 1, rgbCallback);
	depth_sub = it.subscribe("/camera/depth_registered/image_raw", 1, depthCallback);
    while (ros::ok()) { // alive
         if (rgbMat.rows != 0 && depthMat.rows != 0) {
            cv::imshow("rgb", rgbMat);
            cv::imshow("depth", depthMat);
            cv::waitKey(10);
        }
	   
	    
		
	    
		
	    
	    ros::spinOnce();
	}
    return 0;
}

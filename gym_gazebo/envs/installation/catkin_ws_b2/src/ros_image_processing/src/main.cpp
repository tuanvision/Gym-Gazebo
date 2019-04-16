#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include <iostream>
#include <sstream>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ctime>
#include <cstdio>
#include <sstream>	
#include <iostream>

#define RAW_WIDTH 320
#define RAW_HEIGHT 240

#define DEBUG

/* **********************************

Global variables
	rgbMat : rgb image from camera
	depthMat : depth image from camera
	ReturnAngle : angle error, send to ros_drive_cars

********************************** */

cv::Mat rgbMat, depthMat;
float ReturnAngle;

/* ***************
	
	rgbCallback: get rgb image from node astra
					format: CV_8UC3

*************** */
void rgbCallback(const sensor_msgs::ImageConstPtr &msg_rgb) 
{

#ifdef DEBUG
	std::cerr << "call rgb " << '\n';
#endif
	cv_bridge::CvImagePtr rgbPtr = cv_bridge::toCvCopy(*msg_rgb, sensor_msgs::image_encodings::TYPE_8UC3);
    rgbMat = rgbPtr->image.clone();
    cv::cvtColor(rgbMat, rgbMat, cv::COLOR_BGR2RGB);
#ifdef DEBUG
    double lastTime = ros::Time::now().toSec();
    std::cerr << "RGB : " << lastTime << '\n';
#endif    
}

/* ***************
	
	depthCallback: get depth image from node astra
					format : CV_32FC1

*************** */
void depthCallback(const sensor_msgs::ImageConstPtr &msg_depth) 
{
#ifdef DEBUG
	std::cerr << "call depth " << '\n';
#endif
	cv_bridge::CvImagePtr depthPtr = cv_bridge::toCvCopy(*msg_depth, sensor_msgs::image_encodings::TYPE_32FC1);
    depthMat = depthPtr->image.clone();
#ifdef DEBUG
    double lastTime = ros::Time::now().toSec();
    std::cerr << "DEPTH : " << lastTime << '\n';
#endif
}

int main(int argc, char **argv)
{

// Init ros environment 
	ros::init(argc, argv, "ros_image_processing_main");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	image_transport::Subscriber rgb_sub, depth_sub;
	ros::Publisher Angle_pub = nh.advertise<std_msgs::Float32>("Angle", 1);
//	ros::Rate loop_rate(100);
	rgb_sub = it.subscribe("/camera/rgb/image_raw", 1, rgbCallback);
	depth_sub = it.subscribe("/camera/depth_registered/image_raw", 1, depthCallback);
	std_msgs::Float32 msg;
	ReturnAngle = 0;
// Finish init 

	while (ros::ok())
  	{
		if (rgbMat.cols > 0 && depthMat.cols > 0)   // Check get image from astra
		{
			//// Edit code at here
		#ifdef DEBUG
			cv::imshow("rgb", rgbMat);
			cv::imshow("depth", depthMat);
			cv::waitKey(5);
		#endif
		}




		/* Send data to ros_drive_car */
		// msg.data = ReturnAngle;
		msg.data = -10 + rand() % 21;
	#ifdef DEBUG
		ROS_INFO("%.2f\n", msg.data);
	#endif
       	Angle_pub.publish(msg);
		/* Done send data */

    	ros::spinOnce();

//    	loop_rate.sleep();
    	
    }


  return 0;
}

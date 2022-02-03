#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{


// basic
    ros::init(argc,argv, "motor_pub");
    ros::NodeHandle nh;
	
	cv::VideoCapture vc(0);
	if (!vc.isOpened()) return -1;

	vc.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	vc.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	cv::Mat img;

    ros::Publisher pub_dc=nh.advertise<std_msgs::Float32>("pub_dc",10);
    ros::Publisher pub_servo=nh.advertise<std_msgs::Float32>("pub_servo",10);

	ros::Rate loop_rate(10000);

    float dc = 0;
    float servo=90;

    std_msgs::Float32 dc_msg;
    std_msgs::Float32 servo_msg;
	
	double ini_time = ros::Time::now().toSec();

double error = 0;
double error_previous = 0;

double angle_goal=-71;
double aim = -71;

double PID_last = 0;

int adjust_red;
int red_stack=0;
int stop_stack=10;
int green_stack = 0;
int servo_previous = -71;

Scalar lower_w(25,0,210);//0,0,150
Scalar upper_w(255,50,255);

while(ros::ok())
    {
		double time = ros::Time::now().toSec() - ini_time;		vc >> img;
		if(img.empty()) break;
		cv::imshow("cam",img);
		if (cv::waitKey(10)==27) {
			dc_msg.data=0;
        	servo_msg.data=90;
			pub_dc.publish(dc_msg);
        	pub_servo.publish(servo_msg);
			break;
		}

   // Imgae Processing

Rect first_rect(0, img.rows/2, img.cols*2/3, img.rows/2);
	Mat first = img(first_rect);
Mat hsv;
	cvtColor(first,hsv,CV_BGR2HSV);
	
	Scalar lowery(15, 50, 120);
	Scalar uppery(60, 255, 255);	

Mat masky;

inRange(hsv,lowery,uppery,masky);


	threshold(masky,masky,120,255,THRESH_BINARY);

	int y1_stack=6;
	int angle1_y,angle1_x;
	for(int y= masky.rows-1;y>0;y--){
		for(int x=masky.cols-1;x>0;x--){
			if(masky.at<uchar>(y,x)==255){
				y1_stack += 1;
				if(y1_stack==10){
				angle1_y=y;
				angle1_x=x;
				break;
				}
				else
					continue;
			}
		}
		if(y1_stack==10)
			break;
	}
	y1_stack=0;	


int y2_stack=6;
	int angle2_y,angle2_x;
	for(int y= masky.rows-10;y>0;y--){
		for(int x=masky.cols-1;x>0;x--){
			if(masky.at<uchar>(y,x)==255){
				y2_stack += 1;
				if(y2_stack==10){
				angle2_y=y;
				angle2_x=x;
				break;
				}
				else
					continue;
			}
		}
		if(y2_stack==10)
			break;
	}
	y2_stack=0;	
	
	line(first,Point(angle1_x,angle1_y),Point(angle2_x,angle2_y),Scalar(255,0,0),5);

double dx1 = angle2_x - angle1_x;
double dy1 = angle2_y - angle1_y;

	double angle_input;

	angle_input = atan(dy1/dx1)*(180/CV_PI);

angle_input = ceil(angle_input);

imshow("line",first);

// situation

if(abs(angle_goal-angle_input)>50&&abs(angle_goal-angle_input)<100)
angle_input= angle_goal;

else if(abs(angle_goal-angle_input)>100&&angle_goal*angle_input<0)
{
if(angle_input < 0)
angle_input= angle_input +180;
else if(angle_input>0)
angle_input = angle_input - 180;
}
angle_goal = angle_input;


//traffic
Mat Detect_all=img.clone();
Mat Detect_traffic=img.clone();
Mat gray_traffic;
cvtColor(img,gray_traffic,CV_BGR2GRAY);

Mat binary_traffic, thresh_adap, thresh_final;

threshold(gray_traffic,binary_traffic,100,255, THRESH_BINARY_INV);

adaptiveThreshold(gray_traffic,thresh_adap,255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,3,5);

bitwise_and(binary_traffic, thresh_adap, thresh_final);

Mat canny_traffic;
Canny(thresh_final,canny_traffic,100,300,3);

vector<vector<Point> > contours_traffic;
vector<Vec4i> hierarchy;
findContours(canny_traffic,contours_traffic,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point());
vector<vector<Point> > contours_ALL(contours_traffic.size());
vector<Rect> boundRect(contours_traffic.size());

for(int i=0; i<contours_traffic.size();i++){
approxPolyDP(Mat(contours_traffic[i]),contours_ALL[i],1,true);
boundRect[i] = boundingRect(Mat(contours_ALL[i]));
}

vector<Rect> Rect_refined(contours_traffic.size());
int refinerycount =0, refine_constant=0;
Mat refine_Gray, before_refine;
Rect Rect_final;
double ratio;

for(int i=0; i<contours_traffic.size();i++){
ratio = (double)boundRect[i].height/boundRect[i].width;
before_refine = Detect_all(boundRect[i]);
cvtColor(before_refine,refine_Gray,CV_BGR2GRAY);
threshold(refine_Gray,refine_Gray,150,255,THRESH_BINARY);

for(int y=0;y<before_refine.rows;y++){
for(int x=0;x<before_refine.cols;x++){
if(refine_Gray.at<uchar>(y,x)==0)
refine_constant+=1;
}
}
if(refine_constant>130){
        if((ratio <= 0.4) && (ratio >= 0.3) && (boundRect[i].area() <= 5000)&& (boundRect[i].area() >=1500)){

            drawContours(Detect_all, contours_traffic, i, Scalar(0,255,255), 1, 8, hierarchy, 0, Point());
            rectangle(Detect_all, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 1, 8, 0);
Rect_final = boundRect[i];
}
}
refine_constant=0;
}

Mat traffic=Detect_all(Rect_final);

imshow("traffic",Detect_all);


//license plate
Mat Detect_number_all=img.clone();
Mat Detect_number=img.clone();
Mat gray_number;
cvtColor(img,gray_number,CV_BGR2GRAY);

Mat binary_number, thresh_number_adap, thresh_number_final;
threshold(gray_number,binary_number,100,255, THRESH_BINARY_INV);

adaptiveThreshold(gray_number,thresh_number_adap,255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,3,5);

bitwise_and(binary_number, thresh_number_adap, thresh_number_final);

Mat canny_number;
Canny(thresh_number_final,canny_number,100,300,3);

vector<vector<Point> > contours_number;
vector<Vec4i> hierarchy_number;
findContours(canny_number,contours_number,hierarchy_number,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point());
vector<vector<Point> > contours_number_ALL(contours_number.size());
vector<Rect> boundRect_number(contours_number.size());

for(int i=0; i<contours_number.size();i++){
approxPolyDP(Mat(contours_number[i]),contours_number_ALL[i],1,true);
boundRect_number[i] = boundingRect(Mat(contours_number_ALL[i]));
}

vector<Rect> Rect_number_refined(contours_number.size());
int refinerycount_number =0, refine_number_constant=0;
Mat refine_Gray_number, before_number_refine;
Rect Rect_number_final;
double ratio_number;

for(int i=0; i<contours_number.size();i++){
ratio_number = (double)boundRect_number[i].height/boundRect_number[i].width;
before_number_refine = Detect_number_all(boundRect[i]);
cvtColor(before_number_refine,refine_Gray_number,CV_BGR2HSV);

inRange(refine_Gray_number,lower_w,upper_w,refine_Gray_number);
threshold(refine_Gray_number,refine_Gray_number,100,255,THRESH_BINARY);

for(int y=0;y<before_number_refine.rows;y++){
for(int x=0;x<before_number_refine.cols;x++){
if(refine_Gray_number.at<uchar>(y,x)==255)
refine_number_constant+=1;
}
}
if(refine_number_constant>10){
        if((ratio_number <= 0.5) && (ratio_number >= 0.25) && (boundRect_number[i].area() <= 2500)&& (boundRect_number[i].area() >=1000)){

            drawContours(Detect_number_all, contours_number, i, Scalar(0,255,255), 1, 8, hierarchy_number, 0, Point());
            rectangle(Detect_number_all, boundRect_number[i].tl(), boundRect_number[i].br(), Scalar(0,255,0), 1, 8, 0);
Rect_number_final = boundRect_number[i];
}
}
refine_number_constant=0;
}

if(!(Detect_number_all(Rect_number_final).empty()))
{
Mat number_final=Detect_number_all(Rect_number_final);
Mat gray_plate;
cvtColor(number_final,gray_plate, CV_BGR2GRAY);

imwrite("/home/capstone/PLATE.JPG",gray_plate);

imshow("number",Detect_number_all);
}


//pid
double Pterm, Iterm, Dterm;;
float PIDterm;


double Kp = 1.02;
double Kd = 0.00;
double dt=0.01;

double error_previous = error;
double error = aim - angle_input; 


Pterm = Kp*error;
/*if(Pterm>100)
Pterm = 100;
if(Pterm <-100)
Pterm = -100;
*/

PIDterm = Pterm;

if(PIDterm>45)
PIDterm = 45;

if(PIDterm <-45)
PIDterm = -45;

servo = 90 - PIDterm;
dc_msg.data=100;

if(isnan(servo)) servo = servo_previous;

servo_previous = servo;


//red light
Mat traffic_hsv;
Mat light_gray;
cvtColor(traffic,traffic_hsv,CV_BGR2HSV);

Mat range_red1,range_red2,red;
Scalar lower_red1(0,128,128);
Scalar upper_red1(15,255,255);

Scalar lower_red2(150,100,120);
Scalar upper_red2(180,255,255);

inRange(traffic_hsv,lower_red1,upper_red1,range_red1);
inRange(traffic_hsv,lower_red2,upper_red2,range_red2);

bitwise_or(range_red1, range_red2, red);
threshold(red,red,120,255,THRESH_BINARY);

int adjust_red=3;
for(int y=0; y<red.rows;y++){
for(int x=0; x<red.cols;x++){
if(red.at<uchar>(y,x)==255)
adjust_red+=1;
}}

if(adjust_red>5){
printf("RED\n");
red_stack +=1;
Mat stop_before = img.clone();
Mat hsv2;
Rect stop_rect(0,img.rows/2, img.cols*3/5, img.rows/2);

Mat stop = stop_before(stop_rect);

	cvtColor(stop,hsv2,CV_BGR2HSV);


	
	Mat mask_w;
	inRange(hsv2,lower_w,upper_w,mask_w);//first<->hsv2
	threshold(mask_w,mask_w,120,255,THRESH_BINARY);
int w_stack=0;
int green_stack =0;
		for(int x=mask_w.cols-1;x>0;x--){
			if(mask_w.at<uchar>(0,x)==255){
				w_stack += 1;
				while(w_stack> 50){
for(int j = mask_w.rows;j>mask_w.rows*2/3;j--){
for(int i = 0; i < mask_w.cols ; i++){
if(mask_w.at<uchar>(j,i)==255){
	stop.at<Vec3b>(j,i)[0] = 255;
	stop.at<Vec3b>(j,i)[1] = 0;
stop.at<Vec3b>(j,i)[2] = 255;

//green light
Mat traffic_hsv_green;
cvtColor(traffic,traffic_hsv_green,CV_BGR2HSV);

Mat range_green;
Scalar lower_green(50,90,128);
Scalar upper_green(80,255,255);

inRange(traffic_hsv_green,lower_green,upper_green,traffic_hsv_green);
threshold(traffic_hsv_green,traffic_hsv_green,120,255,THRESH_BINARY);

for(int y=0; y<traffic_hsv_green.rows;y++){
for(int x=0; x<traffic_hsv_green.cols;x++){
if(traffic_hsv_green.at<uchar>(y,x)==255)
green_stack+=1;
}
}
if(green_stack>30)
break;

}
}
	w_stack=0;
}
}
}
}
}

if(red_stack>3) 
red_stack = 10;

servo_msg.data =servo;

if(servo<80||servo>100) dc_msg.data = 110;


if (dc<0) dc = -dc;
		else if (dc>255) dc = 255;


if(red_stack>=15) dc_msg.data =0;

if(green_stack>30){
red_stack = 0;
dc_msg.data = 110;
}


		if (servo<40) servo = 45;
		else if (servo>135) servo = 135;
servo_msg.data =servo;
pub_servo.publish(servo_msg);
			pub_dc.publish(dc_msg);

ROS_INFO("angle : %lf error:%lf servo: %lf P:%lf PID: %lf", angle_input, error, servo, Pterm, PIDterm);

loop_rate.sleep();
}
	cv::destroyAllWindows();

	return 0;
}
            

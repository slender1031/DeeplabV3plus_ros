#include "torch/script.h"
#include "torch/torch.h"

#include <iostream>
#include <string>
#include <chrono>
#include <stdio.h>

#include <algorithm>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "std_msgs/String.h"
#include "ros/ros.h"

using namespace std;
using namespace cv;


torch::jit::script::Module module;
torch::Device device = torch::Device(torch::kCUDA);

ros::Publisher image_pub;
cv::Mat label_colors;


cv::Mat GetSegmentation(cv::Mat img) {
    cv::Mat input_image=img;
    //cv::cvtColor(img, input_image, cv::COLOR_BGR2RGB);
    cv::Size picSize = input_image.size();
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);

    torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols,3}, torch::kFloat32).to(device);

    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
    tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
    tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
    //std::cout<<"preprocess finish" << "\n";

    tensor_image=tensor_image.to(torch::kCUDA);
    torch::Tensor output = module.forward({tensor_image}).toTensor();
    //std::cout<<"predict finish"<<"\n";

    output = output.argmax(1).squeeze(0).to(torch::kU8).to(torch::kCPU);
    cv::Mat output_image(picSize, CV_8U, output.data_ptr());

    //cv::Mat mask = cv::Mat(input_image.rows, input_image.cols, CV_8UC1);
    //memcpy(mask.data, output.data_ptr(), output.numel()*sizeof(torch::kU8));

    //cv::Mat res;
    //input_image.copyTo(res, mask);

    return output_image;
}


void img_callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr  cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::RGB8);
    
    cv::Mat image_resize;
    cv::resize(cv_ptr->image, image_resize, cv::Size(640,480), 0, 0, cv::INTER_LINEAR);

   // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cv::Mat segMask = GetSegmentation(image_resize);
    
    //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    //cout<< ttrack << endl;

    cv::Mat mask_color = segMask.clone();
    cv::Mat mask_final;

    cv::Mat mask_color2;
    cv::cvtColor(mask_color, mask_color2, CV_GRAY2RGB);

    LUT(mask_color2, label_colors, mask_final);

   // imshow("mask_final", mask_final);
    //cvWaitKey(0);
    

    cv::Mat out;
    addWeighted(image_resize,0.7,mask_final,0.7, 3, out);
 
    cv_bridge::CvImage  cvi;
    sensor_msgs::Image  ros_img;
    cvi.header.stamp = cv_ptr->header.stamp;
    cvi.header.frame_id = "image";
    cvi.encoding = "rgb8";
    cvi.image = out;
    cvi.toImageMsg(ros_img);
    image_pub.publish(cvi.toImageMsg());
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "img_tran");
    ros::NodeHandle n;

    /***if (argc != 2)
    {
        cerr << "Usage: ./example model_path image_path" << endl;
        return 1;
    }***/

    //const string model_dir=argv[1];
    const string model_dir="/home/slender/torch_ws/deeplab_test_ros/DeeplabV3plus.pt";
    //const string image_path =argv[2];
    const string label_colors_path = "/home/slender/torch_ws/deeplab_test_ros/pascal.png";
    label_colors = cv::imread(label_colors_path, 1);

    //cuda or cpu
    std::cout << "loading segmentation model..." << std::endl;
    try {
        module = torch::jit::load(model_dir);
    }
    catch (const c10::Error& ) {
        std::cerr << "error loading the model\n";
    }
    if (torch::cuda::is_available())
    {
        std::cout << "cuda support: true, now device is GPU" << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    else
    {
        std::cout << "cuda support: false, now device is CPU" << std::endl;
        device = torch::Device(torch::kCPU);
    }
    module.to(device);


    ros::Subscriber sub = n.subscribe("/hikrobot_camera/rgb", 1000, img_callback);
    image_pub = n.advertise<sensor_msgs::Image>("/feature_img", 1000);
    
    ros::spin();
    return 0;


    
    
    /***

  cout<<"loading image......"<<endl;
  cv::Mat img = cv::imread(image_path,CV_LOAD_IMAGE_UNCHANGED);
  //cv::resize(img, img, cv::Size(480,320));
  cv::imshow("img", img);

  //! 计算耗时
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  cv::Mat segMask = GetSegmentation(img);

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  
  double ttrack= std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
  cout<< ttrack << endl;


  //cout << "\n<----------------LUT---------------------->" << endl;
  cv::Mat label_colors = cv::imread(label_colors_path, 1);
  cv::Mat mask_color = segMask.clone();
  cv::Mat mask_final;

  cvtColor(mask_color, mask_color, CV_GRAY2RGB);
  cvtColor(label_colors, label_colors, CV_BGR2RGB);

  LUT(mask_color, label_colors, mask_final);
  //imshow("mask", segMask);
  //imshow("mask_color", mask_color);
  
  imshow("mask_final", mask_final);
  cvWaitKey(0);

  return 0;

  ***/


}

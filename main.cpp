#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
 
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
 
int nfeatures = 100;
Mat src1,src2;
void callback(int, void*);
int main(int arc, char** argv) { 
	src1 = imread("pictures/left.png");
	src2 = imread("pictures/right.png");
	namedWindow("output",0);
	//resizeWindow("output", 800, 400);//重置窗口大小
	createTrackbar("nfeatures", "output", &nfeatures,500, callback);;
	callback(0, 0);
	waitKey(0);
	return 0;
} 
void callback(int, void*) {
	//实例化一个SIFT检测类对象的结构指针
	Ptr<SIFT>sift = SIFT::create(nfeatures);
	vector<KeyPoint>keypoints1, keypoints2;
	//检测关键点
	sift->detect(src1, keypoints1);
	sift->detect(src2, keypoints2);
 
	printf("total keypoints1:%d\n", (int)keypoints1.size());
	printf("total keypoints2:%d\n", (int)keypoints2.size());
 
	//绘制关键点
	Mat keypoints1_img, keypoints2_img;
	drawKeypoints(src1, keypoints1, keypoints1_img);
	drawKeypoints(src2, keypoints2, keypoints2_img);
	imshow("keypoints1_img", keypoints1_img);
	imshow("keypoints2_img", keypoints2_img);
 
	//计算描述子即特征向量
	Mat discriptions1, discriptions2;
	sift->compute(src1,keypoints1, discriptions1);
	sift->compute(src2,keypoints2, discriptions2);
 
	//实例化一个匹配的对象
	Ptr<DescriptorMatcher>matcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch>matches;
	matcher->match(discriptions1, discriptions2, matches);
 
	//绘制匹配图
	Mat match_img;
	drawMatches(src1, keypoints1, src2, keypoints2, matches,match_img);
	imshow("output", match_img);
}

/*
int main()
{
   vector<string> msg {"Hello", "C++", "World", "from", "VS Code!", "and the C++ extension!"};

   for (const string& word : msg)
   {
      cout << word << " ";
   }
   cout << endl;
}
*/
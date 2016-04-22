#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define FIRST 1
#define BLACK 0
#define RED 0
#define GREEN 2
#define BLUE 1
#define SURREND 0
#define ITERATOR 0
#define SPLIT 0
#define ADD 0
#define ROI 0
#define SIZE 0
#define LUTT 0
#define THRESHOLDED 1
#define ERODE 1
#define HIST 1
#define BLUR 1
#define HIGHT 1




int main()
{
#if FIRST
	const char *fileNameWrite = "d://data//b3.jpg";
	const char *fileName = "d://data//b.jpg";
	Mat img = imread(fileName);

	Mat grey;
	cvtColor(img, grey, CV_BGR2GRAY);

	imshow("original", img);
	imshow("grey", grey);
	/*
	Scalar intensity = grey.at<uchar>(Point(111, 243));
	cout << "intensity:" << intensity.val[0] << endl;
	cout << "intensity:" << intensity.val[1] << endl;
	cout << "intensity:" << intensity.val[2] << endl;
	*/
#endif // FIRST	

#if BLACK
	int nl = grey.rows;
	int nc = grey.cols * grey.channels();
	uchar *p = NULL;

	if (grey.isContinuous())
	{
		nc = nc * nl;
		nl = 1;
	}

	for (int i = 0; i < nl; i++)
	{
		p = grey.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{

			//cout << (int)p[j] << "    ";
			p[j] = p[j] / 64 * 64 + 32;
			//cout << (int)p[j] << endl;
		}
	}
	imshow("black", grey);
#endif // BLACK


#if ITERATOR

	MatIterator_<uchar> it = grey.begin<uchar>();
	MatIterator_<uchar> itend = grey.end<uchar>();

	for (; it != itend; it++)
	{
		(*it) = (uchar)0;

	}

	imshow("iterator", grey);

#endif // ITERATOR	

#if RED
	Mat red = img.clone();
	MatIterator_<Vec3b> it2 = red.begin<Vec3b>();
	MatIterator_<Vec3b> itend2 = red.end<Vec3b>();
	while (it2 != itend2)
	{
		(*it2)[0] = 0;
		(*it2)[1] = 0;
		it2++;
	}
	imshow("red", red);
#endif // RED

#if GREEN
	Mat green = img.clone();
	MatIterator_<Vec3b> it3 = green.begin<Vec3b>();
	MatIterator_<Vec3b> itend3 = green.end<Vec3b>();
	while (it3 != itend3)
	{
		(*it3)[0] = 0;
		(*it3)[2] = 0;
		it3++;
	}
	imshow("green", green);
#endif // GREEN

#if BLUE
	Mat blue = img.clone();
	MatIterator_<Vec3b> it4 = blue.begin<Vec3b>();
	MatIterator_<Vec3b> itend4 = blue.end<Vec3b>();
	while (it4 != itend4)
	{
		(*it4)[2] = 0;
		(*it4)[1] = 0;
		it4++;
	}
	imshow("BLUE", blue);
#endif // BLUE

#if SURREND
	Mat surrend = grey.clone();
	int nr = surrend.rows;
	int nc = surrend.cols * surrend.channels();


	uchar *pPre = NULL;
	uchar *p = NULL;
	uchar *pNex = NULL;
	for (int i = 1; i < nr - 1; i++)
	{
		pPre = surrend.ptr<uchar>(i - 1);
		p = surrend.ptr<uchar>(i);
		pNex = surrend.ptr<uchar>(i + 1);
		for (int j = 1; j < nc - 1; j++)
		{
			p[j] = 5 * p[j] - p[j - 1] - p[j + 1] + pPre[j] + pNex[j];
		}
	}
	imshow("surrend", surrend);

#endif 

#if SPLIT

	//����ɫͼ�������ͨ����ȡ�����ֱ�������֮���ٺ�������ÿһ��ͨ������һ��ͼƬ

	Mat splitTest = img.clone();
	vector<Mat> planes;
	cv::split(splitTest, planes);
	imshow("planes[0]", planes[0]);
	imshow("planes[1]", planes[1]);
	imshow("planes[2]", planes[2]);

	planes[1] = grey;		//�����grey�Ǹ��Ҷ�ͼ�����Ҷ�ͼ�����ɫͨ�����������

	merge(planes, splitTest);
	imshow("splitTest", splitTest);

#endif //SPLIT

#if ADD
	Mat img2 = imread("d://data//timg.jpg", 0);
	Mat add = img.clone();
	Mat temp;
	cv::add(grey, cv::Scalar(100), temp);
	//cv::add(grey, grey, add);	//���������������Ҷ�ͼ����ӡ�
	//cv::addWeighted(grey,2, grey,3,20, add);	//����ǽ�����ͼ����Ȩ�ص���
	cv::subtract(temp, grey, add);	//����ͼ����Ԫ������������������Ȳ������������������ͼ���С������ͬ������������ĵ�����һ����ĸ����

	imshow("add", add);
#endif

#if ROI
	Mat image = imread("d://data//b.jpg");
	Mat logo = imread("d://data//log.png");
	cout << "image:" << image.cols << " " << image.rows << endl;
	cout << " logo:" << logo.cols << " " << logo.rows << endl;
	Mat roi;
	roi = image(Rect(160, 160, logo.cols, logo.rows));		//���ﲻ��Ϥ
	addWeighted(roi, 1, logo, 0.3, 0., roi);
	cout << " roi:" << roi.cols << " " << roi.rows << endl;
	imshow("roi", roi);
	imshow("logo", logo);

#endif // ROI

#if SIZE
	Mat size;
	img = imread("d://data//size.png");
	resize(img, size, Size(img.cols / 3, img.rows / 3));
	imshow("size1", size);

	resize(img, size, Size(img.cols * 2, img.rows * 2));
	imshow("size2", size);


#endif // SIZE

#if LUTT
	Mat result;
	int dim(256);
	Mat lut(1, &dim, CV_8U);
	for (int i = 0; i < 256; ++i)
		lut.at<uchar>(i) = 255 - i;		//����atҲ���Է������أ�Ϊ����255-i��
	LUT(grey, lut, result);
	imshow("lut", lut);

#endif // LUTT

#if THRESHOLDED
	Mat thresholded;
	threshold(grey, thresholded, 210, 255, THRESH_BINARY);
	imshow("thresholded1", thresholded);

	threshold(grey, thresholded, 210, 255, THRESH_TRUNC);
	imshow("thresholded2", thresholded);

#endif // THRESHOLDED

#if ERODE
	Mat x(5, 5, CV_8U, Scalar(0));
	//cout << "x:" << x.cols << "  " <<  x.rows << endl;
	for (int i = 0; i < 5; i++)
	{
		x.at<uchar>(i, i) = 1;
		x.at<uchar>(4 - i, i) = 1;
	}


	grey = imread("d://data//erode.jpg", 0);
	imshow("grey", grey);

	Mat result;
	Mat element(5, 5, CV_8U, Scalar(1));		//���ڸ�ʴ�����͵ĽṹԪ�أ��������Լ�������Ҫ����״��

	dilate(grey, result, element);
	imshow("dilate", result);

	erode(result, result, element);				//����������λ�ô���
	imshow("erode", result);

	//ʹ�ýṹԪ��B��A���п�����������B��A��ʴ��Ȼ������B�Խ���������͡�
	//ʹ�ýṹԪ��B��A�ıղ���������B��A�������ͣ�Ȼ����B�Խ�����и�ʴ��

	Mat close;
	morphologyEx(grey, close, MORPH_CLOSE, element);
	imshow("close", close);


#endif // ERODE

#if HIST
	Mat hist;
	grey = imread("d://data//a.jpg", 0);
	imshow("grey", grey);
	equalizeHist(grey, hist);
	imshow("hist", hist);
#endif // HIST

#if BLUR
	Mat blurr;
	blur(grey, blurr, Size(3, 3));			//�ռ��ͨ��ֵ�˲���ԭ����ʲô��
	imshow("blur", blurr);

	GaussianBlur(grey, blurr, Size(5, 5), 1.5);	//��˹ģ����ԭ��
	imshow("gaussian", blurr);

	pyrDown(grey, blurr);
	imshow("pyrDown", blurr);				//�²�������������СͼƬ����֮ǰ��resize()�к�����

	pyrUp(grey, blurr);						//�ϲ�����
	imshow("pyrUp", blurr);

#endif // BLUR

#if HIGHT
	Mat sobel;
	Sobel(grey, sobel, CV_8U, 0, 1, 3, 0.4, 128);
	imshow("sobelY", sobel);

	Sobel(grey, sobel, CV_8U, 1, 0, 3, 0.4, 128);	//Sobel �˲�����������ʲô�ģ�
	imshow("sobelX", sobel);


#endif // HIGHT
	imwrite(fileNameWrite, grey);
	waitKey();


	//system("pause");
	return 0;
}

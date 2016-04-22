#pragma once
#include <opencv2/opencv.hpp>

class LaplacianZC {
private:
	// orignal image
	cv::Mat img;
	// 32-bit float image containing the Laplacian
	cv::Mat laplace;
	// Aperture size of the laplacian kernel
	int aperture;
public:
	LaplacianZC() : aperture(3) {}
	// Set the aperture size of the kernel
	void setAperture(int a) {
		aperture = a;
	}
	// Compute the floating point Laplacian
	cv::Mat computeLaplacian(const cv::Mat &image) {
		// Compute Laplacian
		cv::Laplacian(image, laplace, CV_32F, aperture);
		// Keep local copy of the image
		// (used for zero-crossings)
		img = image.clone();
		return laplace;
	}
	// Get the Laplacian result in 8-bit image
	// zero corresponds to gray level 128
	// if no scale is provided, then the max value will be
	// scaled to intensity 255
	// You must call computeLaplacian before calling this
	cv::Mat getLaplacianImage(double scale = -1.0) {
		if (scale < 0) {
			double lapmin, lapmax;
			cv::minMaxLoc(laplace, &lapmin, &lapmax);
			scale = 127 / std::max(-lapmin, lapmax);
		}
		cv::Mat laplaceImage;
		laplace.convertTo(laplaceImage, CV_8U, scale, 128);
		return laplaceImage;
	}
};

class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	void setMarkers(const cv::Mat& markerImage) {
		// Convert to image of ints
		markerImage.convertTo(markers, CV_32S);
	}
	cv::Mat process(const cv::Mat &image) {
		// Apply watershed
		cv::watershed(image, markers);
		return markers;
	}
};
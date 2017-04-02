#pragma once
#include <vector>

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

void generateRandomPoints(std::vector<cv::Point2d>& points,
                          int num_points,
                          const cv::Point2d& max_point,
                          const cv::Point2d& min_point = cv::Point2d(0,0));

void generatePointColors(const std::vector<cv::Point2d>& points,
                         std::vector<cv::Vec3b>& colors);

enum class DistanceMetric {
  NormL2 = 0,
  NormL1,
  NormL1MinSide,
  NormL1MaxSide
};

cv::Mat3b drawVoronoi(const std::vector<cv::Point2d>& points,
                      const std::vector<cv::Vec3b>& colors,
                      const cv::Size& size,
                      DistanceMetric distance_metric);

void applyGaussianBlurIterations(cv::Mat3b& image, int kernel_width, int iterations);

void drawVoronoiCenters(cv::Mat3b& image, const std::vector<cv::Point2d>& points, int radius);

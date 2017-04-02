#include "voronoi.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

class ScopedTimer {
 public:
  ScopedTimer(const std::string& label)
    : label_(label),
      start_time_(cv::getTickCount()) {
  }

  ~ScopedTimer() {
    const auto end_time = cv::getTickCount();
    std::cout << label_ << " took " << 1000.0 * static_cast<double>(end_time - start_time_) / cv::getTickFrequency() << " ms" << std::endl;
  }

 private:
  const std::string label_;
  const int64_t start_time_;
};


void generateRandomPoints(std::vector<cv::Point2d>& points,
                          int num_points,
                          const cv::Point2d& max_point,
                          const cv::Point2d& min_point) {
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis_color(0, 255);

    std::uniform_real_distribution<double>  dis_x(min_point.x, max_point.x);
    std::uniform_real_distribution<double>  dis_y(min_point.y, max_point.y);
    //std::normal_distribution<double>  dis_x(0.0, width);
    //std::normal_distribution<double>  dis_y(0.0, height);

    for (int i=0; i<num_points; ++i) {
      points.emplace_back(dis_x(gen), dis_y(gen));
    }
  }
}

void generatePointColors(const std::vector<cv::Point2d>& points,
                         std::vector<cv::Vec3b>& colors) {
  {
    cv::Mat3b colormap;
    {
      cv::Mat1b tmp(1, 256);
      for (int i=0; i<tmp.cols; ++i)
        tmp(i) = static_cast<uint8_t>(i);
      cv::applyColorMap(tmp, colormap, cv::COLORMAP_JET);
    }

    for (size_t i=0; i<points.size(); ++i) {
      const int colormap_idx = cv::saturate_cast<uint8_t>(((1.0 * i) / (1.0 * points.size())) * 255.0);
      colors.emplace_back(colormap(colormap_idx));
    }
  }
}

template <class NormOp>
class VoronoiParallel : public cv::ParallelLoopBody {
 public:
  VoronoiParallel(const std::vector<cv::Point2d>& points,
                  const std::vector<cv::Vec3b>& colors,
                  cv::Mat3b& image)
      : points_(points),
        colors_(colors),
        image_(image) {
  }

 private:
  void operator()(const cv::Range& range) const override {
    for (int r = range.start; r < range.end; ++r) {
      const int row = r / image_.cols;
      const int col = r % image_.cols;
      size_t best_idx = std::numeric_limits<size_t>::max();
      double best_norm = std::numeric_limits<double>::max();
      for (size_t i=0; i<points_.size(); ++i) {
        const cv::Point2d vec = cv::Point2d(col, row) - points_[i];
        const double norm = NormOp()(vec);
        if (norm < best_norm) {
          best_norm = norm;
          best_idx = i;
        }
      }
      image_(row, col) = colors_[best_idx];
    }
  }

  const std::vector<cv::Point2d>& points_;
  const std::vector<cv::Vec3b>& colors_;
  cv::Mat3b& image_;
};

template <class NormOp>
void drawVoronoiImpl(const std::vector<cv::Point2d>& points,
                     const std::vector<cv::Vec3b>& colors,
                     cv::Mat3b& image) {
  VoronoiParallel<NormOp> vp(points, colors, image);
  cv::parallel_for_(cv::Range(0, image.cols*image.rows), vp);
}

struct NormL2 {
  inline
  double operator()(const cv::Point2d& vec) {
    return std::sqrt(vec.x*vec.x + vec.y*vec.y);
  }
};

struct NormL1 {
  inline
  double operator()(const cv::Point2d& vec) {
    return std::abs(vec.x) + std::abs(vec.y);
  }
};

struct NormL1MinSide {
  inline
  double operator()(const cv::Point2d& vec) {
    return std::min(std::abs(vec.x), std::abs(vec.y));
  }
};

struct NormL1MaxSide {
  inline
  double operator()(const cv::Point2d& vec) {
    return std::max(std::abs(vec.x), std::abs(vec.y));
  }
};

cv::Mat3b drawVoronoi(const std::vector<cv::Point2d>& points,
                      const std::vector<cv::Vec3b>& colors,
                      const cv::Size& size,
                      DistanceMetric distance_metric) {
  //cv::setNumThreads(0);
  cv::Mat3b image = cv::Mat3b::zeros(size);
  ScopedTimer t("voronoi");
  switch (distance_metric) {
    case DistanceMetric::NormL2:
      drawVoronoiImpl<NormL2>(points, colors, image);
      break;
    case DistanceMetric::NormL1:
      drawVoronoiImpl<NormL1>(points, colors, image);
      break;
    case DistanceMetric::NormL1MinSide:
      drawVoronoiImpl<NormL1MinSide>(points, colors, image);
      break;
    case DistanceMetric::NormL1MaxSide:
      drawVoronoiImpl<NormL1MaxSide>(points, colors, image);
      break;
  }
  return image;
}

void applyGaussianBlurIterations(cv::Mat3b& image, int kernel_width, int iterations) {
  const cv::Size kernel_size(kernel_width, kernel_width);
  ScopedTimer t("Gaussian blur");
  for (int i=0; i<iterations; ++i) {
    cv::GaussianBlur(image, image, kernel_size, 0);
  }
}

void drawVoronoiCenters(cv::Mat3b& image, const std::vector<cv::Point2d>& points, int radius) {
  ScopedTimer("draw centers");
  for (const auto& pt : points) {
    cv::circle(image, pt, radius, cv::Scalar::all(0), -1);
  }
}

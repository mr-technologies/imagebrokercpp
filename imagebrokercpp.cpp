// std
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>

// json
#include <nlohmann/json.hpp>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>
#if defined(HAVE_CUDA) && defined(HAVE_OPENGL)
#include <opencv2/core/opengl.hpp>
#define OPENCV_HAS_CUDA_AND_OPENGL 1
#else
#pragma message("Missing CUDA or OpenGL support in OpenCV, make sure to adjust configuration file accordingly")
#endif

// IFF SDK
#include <iffwrapper.hpp>


constexpr int MAX_WINDOW_WIDTH  = 1280;
constexpr int MAX_WINDOW_HEIGHT = 1024;

namespace iff = iffwrapper;

int main()
{
    std::ifstream cfg_file("imagebrokercpp.json");
    const std::string config_str = { std::istreambuf_iterator<char>(cfg_file), std::istreambuf_iterator<char>() };

    const auto config = nlohmann::json::parse(config_str, nullptr, true, true);
    const auto it_chains = config.find("chains");
    if(it_chains == config.end())
    {
        std::cerr << "Invalid configuration provided: missing `chains` section\n";
        return EXIT_FAILURE;
    }
    if(!it_chains->is_array())
    {
        std::cerr << "Invalid configuration provided: section `chains` must be an array\n";
        return EXIT_FAILURE;
    }
    if(it_chains->empty())
    {
        std::cerr << "Invalid configuration provided: section `chains` must not be empty\n";
        return EXIT_FAILURE;
    }
    const auto it_iff = config.find("IFF");
    if(it_iff == config.end())
    {
        std::cerr << "Invalid configuration provided: missing `IFF` section\n";
        return EXIT_FAILURE;
    }

    iff::initialize_engine(it_iff.value().dump());

    std::vector<std::shared_ptr<iff::chain>> chains;
    for(const auto& chain_config : it_chains.value())
    {
        auto chain = std::make_shared<iff::chain>(chain_config.dump(), [](const std::string& element_id, int error_code)
        {
            std::cerr << "Chain element `" << element_id << "` reported an error: " << error_code;
        });
        chains.push_back(chain);
    }

    const auto& current_chain = chains.front();

#if OPENCV_HAS_CUDA_AND_OPENGL
    using Mat = cv::cuda::GpuMat;
#else
    using Mat = cv::Mat;
#endif
    std::mutex render_mutex;
    Mat render_image;

    current_chain->set_export_callback("exporter", [&](const void* data, size_t size, iff::image_metadata metadata)
    {
        void* const img_data = const_cast<void*>(data);
        Mat src_image(cv::Size(metadata.width, metadata.height), CV_8UC4, img_data, metadata.width * 4 + metadata.padding);
        std::lock_guard<std::mutex> render_lock(render_mutex);
        src_image.copyTo(render_image);
    });

    current_chain->execute(nlohmann::json{ { "exporter", { { "command", "on" } } } }.dump());

    const std::string window_name = "IFF SDK Image Broker Sample";
    bool size_set = false;
#if OPENCV_HAS_CUDA_AND_OPENGL
    cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_OPENGL);
    cv::setWindowProperty(window_name, cv::WND_PROP_VSYNC, 1);
    cv::ogl::Texture2D tex;
#else
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
#endif

    iff::log(iff::log_level::info, "imagebroker", "Press Esc to terminate the program");
    while(true)
    {
        if((cv::pollKey() & 0xffff) == 27)
        {
            iff::log(iff::log_level::info, "imagebroker", "Esc key was pressed, stopping the program");
            break;
        }
        std::lock_guard<std::mutex> render_lock(render_mutex);
        if(!render_image.empty())
        {
            if(!size_set)
            {
                auto size = render_image.size();
                if(size.width > MAX_WINDOW_WIDTH)
                {
                    size.height = static_cast<decltype(size)::value_type>(MAX_WINDOW_WIDTH / size.aspectRatio());
                    size.width = MAX_WINDOW_WIDTH;
                }
                if(size.height > MAX_WINDOW_HEIGHT)
                {
                    size.width = static_cast<decltype(size)::value_type>(MAX_WINDOW_HEIGHT * size.aspectRatio());
                    size.height = MAX_WINDOW_HEIGHT;
                }
                cv::resizeWindow(window_name, size);
                size_set = true;
            }
#if OPENCV_HAS_CUDA_AND_OPENGL
            tex.copyFrom(render_image);
            cv::imshow(window_name, tex);
#else
            cv::imshow(window_name, render_image);
#endif
        }
    }

    chains.clear();

    iff::finalize_engine();

    return EXIT_SUCCESS;
}

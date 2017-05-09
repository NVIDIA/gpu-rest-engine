#include "classification.h"

#include <iosfwd>
#include <vector>

#define USE_CUDNN 1
#include <caffe/caffe.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "gpu_allocator.h"

using namespace caffe;
using std::string;
using GpuMat = cv::cuda::GpuMat;
using namespace cv;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

/* Based on the cpp_classification example of Caffe, but with GPU
 * image preprocessing and a simple memory pool. */
class Classifier
{
public:
    Classifier(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               const string& label_file,
               GPUAllocator* allocator);

    std::vector<Prediction> Classify(const Mat& img, int N = 5);

private:
    void SetMean(const string& mean_file);

    std::vector<float> Predict(const Mat& img);

    void WrapInputLayer(std::vector<GpuMat>* input_channels);

    void Preprocess(const Mat& img,
                    std::vector<GpuMat>* input_channels);

private:
    GPUAllocator* allocator_;
    std::shared_ptr<Net<float>> net_;
    Size input_geometry_;
    int num_channels_;
    GpuMat mean_;
    std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file,
                       GPUAllocator* allocator)
    : allocator_(allocator)
{
    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    net_ = std::make_shared<Net<float>>(model_file, TEST);
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file);

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
        << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const Mat& img, int N)
{
    std::vector<float> output = Predict(img);

    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i)
    {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i)
    {
        /* Extract an individual channel. */
        Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    Mat packed_mean;
    merge(channels, packed_mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    Scalar channel_mean = mean(packed_mean);
    Mat host_mean = Mat(input_geometry_, packed_mean.type(), channel_mean);
    mean_.upload(host_mean);
}

std::vector<float> Classifier::Predict(const Mat& img)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<GpuMat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate GpuMat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<GpuMat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_gpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        GpuMat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const Mat& host_img,
                            std::vector<GpuMat>* input_channels)
{
    GpuMat img(host_img, allocator_);
    /* Convert the input image to the input image format of the network. */
    GpuMat sample(allocator_);
    if (img.channels() == 3 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    GpuMat sample_resized(allocator_);
    if (sample.size() != input_geometry_)
        cuda::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    GpuMat sample_float(allocator_);
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    GpuMat sample_normalized(allocator_);
    cuda::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the GpuMat
     * objects in input_channels. */
    cuda::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->gpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

/* By using Go as the HTTP server, we have potentially more CPU threads than
 * available GPUs and more threads can be added on the fly by the Go
 * runtime. Therefore we cannot pin the CPU threads to specific GPUs.  Instead,
 * when a CPU thread is ready for inference it will try to retrieve an
 * execution context from a queue of available GPU contexts and then do a
 * cudaSetDevice() to prepare for execution. Multiple contexts can be allocated
 * per GPU. */
class ExecContext
{
public:
    friend ScopedContext<ExecContext>;

    static bool IsCompatible(int device)
    {
        cudaError_t st = cudaSetDevice(device);
        if (st != cudaSuccess)
            return false;

        cuda::DeviceInfo info;
        if (!info.isCompatible())
            return false;

        return true;
    }

    ExecContext(const string& model_file,
                 const string& trained_file,
                 const string& mean_file,
                 const string& label_file,
                 int device)
        : device_(device)
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");

        allocator_.reset(new GPUAllocator(1024 * 1024 * 128));
        caffe_context_.reset(new Caffe);
        Caffe::Set(caffe_context_.get());
        classifier_.reset(new Classifier(model_file, trained_file,
                                         mean_file, label_file,
                                         allocator_.get()));
        Caffe::Set(nullptr);
    }

    Classifier* CaffeClassifier()
    {
        return classifier_.get();
    }

private:
    void Activate()
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");
        allocator_->reset();
        Caffe::Set(caffe_context_.get());
    }

    void Deactivate()
    {
        Caffe::Set(nullptr);
    }

private:
    int device_;
    std::unique_ptr<GPUAllocator> allocator_;
    std::unique_ptr<Caffe> caffe_context_;
    std::unique_ptr<Classifier> classifier_;
};

struct classifier_ctx
{
    ContextPool<ExecContext> pool;
};

/* Currently, 2 execution contexts are created per GPU. In other words, 2
 * inference tasks can execute in parallel on the same GPU. This helps improve
 * GPU utilization since some kernel operations of inference will not fully use
 * the GPU. */
constexpr static int kContextsPerDevice = 2;

classifier_ctx* classifier_initialize(char* model_file, char* trained_file,
                                      char* mean_file, char* label_file)
{
    try
    {
        ::google::InitGoogleLogging("inference_server");

        int device_count;
        cudaError_t st = cudaGetDeviceCount(&device_count);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not list CUDA devices");

        ContextPool<ExecContext> pool;
        for (int dev = 0; dev < device_count; ++dev)
        {
            if (!ExecContext::IsCompatible(dev))
            {
                LOG(ERROR) << "Skipping device: " << dev;
                continue;
            }

            for (int i = 0; i < kContextsPerDevice; ++i)
            {
                std::unique_ptr<ExecContext> context(new ExecContext(model_file, trained_file,
                                                                       mean_file, label_file, dev));
                pool.Push(std::move(context));
            }
        }

        if (pool.Size() == 0)
            throw std::invalid_argument("no suitable CUDA device");

        classifier_ctx* ctx = new classifier_ctx{std::move(pool)};
        /* Successful CUDA calls can set errno. */
        errno = 0;
        return ctx;
    }
    catch (const std::invalid_argument& ex)
    {
        LOG(ERROR) << "exception: " << ex.what();
        errno = EINVAL;
        return nullptr;
    }
}

const char* classifier_classify(classifier_ctx* ctx,
                                char* buffer, size_t length)
{
    try
    {
        _InputArray array(buffer, length);

        Mat img = imdecode(array, -1);
        if (img.empty())
            throw std::invalid_argument("could not decode image");

        std::vector<Prediction> predictions;
        {
            /* In this scope an execution context is acquired for inference and it
             * will be automatically released back to the context pool when
             * exiting this scope. */
            ScopedContext<ExecContext> context(ctx->pool);
            auto classifier = context->CaffeClassifier();
            predictions = classifier->Classify(img);
        }

        /* Write the top N predictions in JSON format. */
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            Prediction p = predictions[i];
            os << "{\"confidence\":" << std::fixed << std::setprecision(4)
               << p.second << ",";
            os << "\"label\":" << "\"" << p.first << "\"" << "}";
            if (i != predictions.size() - 1)
                os << ",";
        }
        os << "]";

        errno = 0;
        std::string str = os.str();
        return strdup(str.c_str());
    }
    catch (const std::invalid_argument&)
    {
        errno = EINVAL;
        return nullptr;
    }
}

void classifier_destroy(classifier_ctx* ctx)
{
    delete ctx;
}

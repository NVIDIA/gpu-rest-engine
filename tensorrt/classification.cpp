#include "classification.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "NvInfer.h"
#include "NvCaffeParser.h"

#include "common.h"
#include "gpu_allocator.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;
using GpuMat = cuda::GpuMat;
using namespace cv;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};
static Logger gLogger;

class InferenceEngine
{
public:
    InferenceEngine(const string& model_file,
                    const string& trained_file);

    ~InferenceEngine();

    ICudaEngine* Get() const
    {
        return engine_;
    }

private:
    ICudaEngine* engine_;
};

InferenceEngine::InferenceEngine(const string& model_file,
                                 const string& trained_file)
{
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();

    ICaffeParser* parser = createCaffeParser();
    auto blob_name_to_tensor = parser->parse(model_file.c_str(),
                                             trained_file.c_str(),
                                             *network,
                                             nvinfer1::DataType::kFLOAT);
    CHECK(blob_name_to_tensor) << "Could not parse the model";

    // specify which tensors are outputs
    network->markOutput(*blob_name_to_tensor->find("prob"));

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 30);

    engine_ = builder->buildCudaEngine(*network);
    CHECK(engine_) << "Failed to create inference engine.";

    network->destroy();
    builder->destroy();
}

InferenceEngine::~InferenceEngine()
{
    engine_->destroy();
}

class Classifier
{
public:
    Classifier(std::shared_ptr<InferenceEngine> engine,
               const string& mean_file,
               const string& label_file,
               GPUAllocator* allocator);

    ~Classifier();

    std::vector<Prediction> Classify(const Mat& img, int N = 5);

private:
    void SetModel();

    void SetMean(const string& mean_file);

    void SetLabels(const string& label_file);

    std::vector<float> Predict(const Mat& img);

    void WrapInputLayer(std::vector<GpuMat>* input_channels);

    void Preprocess(const Mat& img,
                    std::vector<GpuMat>* input_channels);

private:
    GPUAllocator* allocator_;
    std::shared_ptr<InferenceEngine> engine_;
    IExecutionContext* context_;
    GpuMat mean_;
    std::vector<string> labels_;
    DimsCHW input_dim_;
    Size input_cv_size_;
    float* input_layer_;
    DimsCHW output_dim_;
    float* output_layer_;
};

Classifier::Classifier(std::shared_ptr<InferenceEngine> engine,
                       const string& mean_file,
                       const string& label_file,
                       GPUAllocator* allocator)
    : allocator_(allocator),
      engine_(engine)
{
    SetModel();
    SetMean(mean_file);
    SetLabels(label_file);
}

Classifier::~Classifier()
{
    context_->destroy();
    CHECK_EQ(cudaFree(input_layer_), cudaSuccess) << "Could not free input layer";
    CHECK_EQ(cudaFree(output_layer_), cudaSuccess) << "Could not free output layer";
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

    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i)
    {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}

void Classifier::SetModel()
{
    ICudaEngine* engine = engine_->Get();

    context_ = engine->createExecutionContext();
    CHECK(context_) << "Failed to create execution context.";

    int input_index = engine->getBindingIndex("data");
    input_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(input_index));
    input_cv_size_ = Size(input_dim_.w(), input_dim_.h());
    // FIXME: could be wrapped in a thrust or GpuMat object.
    size_t input_size = input_dim_.c() * input_dim_.h() * input_dim_.w() * sizeof(float);
    cudaError_t st = cudaMalloc(&input_layer_, input_size);
    CHECK_EQ(st, cudaSuccess) << "Could not allocate input layer.";

    int output_index = engine->getBindingIndex("prob");
    output_dim_ = static_cast<DimsCHW&&>(engine->getBindingDimensions(output_index));
    size_t output_size = output_dim_.c() * output_dim_.h() * output_dim_.w() * sizeof(float);
    st = cudaMalloc(&output_layer_, output_size);
    CHECK_EQ(st, cudaSuccess) << "Could not allocate output layer.";
}

void Classifier::SetMean(const string& mean_file)
{
    ICaffeParser* parser = createCaffeParser();
    IBinaryProtoBlob* mean_blob = parser->parseBinaryProto(mean_file.c_str());
    parser->destroy();
    CHECK(mean_blob) << "Could not load mean file.";

    DimsNCHW mean_dim = mean_blob->getDimensions();
    int c = mean_dim.c();
    int h = mean_dim.h();
    int w = mean_dim.w();
    CHECK_EQ(c, input_dim_.c())
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<Mat> channels;
    float* data = (float*)mean_blob->getData();
    for (int i = 0; i < c; ++i)
    {
        /* Extract an individual channel. */
        Mat channel(h, w, CV_32FC1, data);
        channels.push_back(channel);
        data += h * w;
    }

    /* Merge the separate channels into a single image. */
    Mat packed_mean;
    merge(channels, packed_mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    Scalar channel_mean = mean(packed_mean);
    Mat host_mean = Mat(input_cv_size_, packed_mean.type(), channel_mean);
    mean_.upload(host_mean);
}

void Classifier::SetLabels(const string& label_file)
{
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));
}

std::vector<float> Classifier::Predict(const Mat& img)
{
    std::vector<GpuMat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    void* buffers[2] = { input_layer_, output_layer_ };
    context_->execute(1, buffers);

    size_t output_size = output_dim_.c() * output_dim_.h() * output_dim_.w();
    std::vector<float> output(output_size);
    cudaError_t st = cudaMemcpy(output.data(), output_layer_, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess)
        throw std::runtime_error("could not copy output layer back to host");

    return output;
}

/* Wrap the input layer of the network in separate Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<GpuMat>* input_channels)
{
    int width = input_dim_.w();
    int height = input_dim_.h();
    float* input_data = input_layer_;
    for (int i = 0; i < input_dim_.c(); ++i)
    {
        GpuMat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const Mat& host_img,
                            std::vector<GpuMat>* input_channels)
{
    int num_channels = input_dim_.c();
    GpuMat img(host_img, allocator_);
    /* Convert the input image to the input image format of the network. */
    GpuMat sample(allocator_);
    if (img.channels() == 3 && num_channels == 1)
        cuda::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels == 1)
        cuda::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels == 3)
        cuda::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels == 3)
        cuda::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    GpuMat sample_resized(allocator_);
    if (sample.size() != input_cv_size_)
        cuda::resize(sample, sample_resized, input_cv_size_);
    else
        sample_resized = sample;

    GpuMat sample_float(allocator_);
    if (num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    GpuMat sample_normalized(allocator_);
    cuda::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the Mat
     * objects in input_channels. */
    cuda::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == input_layer_)
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

	cuda::DeviceInfo dev_info;
	if (dev_info.majorVersion() < 3)
            return false;

        return true;
    }

    ExecContext(std::shared_ptr<InferenceEngine> engine,
                const string& mean_file,
                const string& label_file,
                int device)
        : device_(device)
    {
	cudaError_t st = cudaSetDevice(device_);

	if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");

        allocator_.reset(new GPUAllocator(1024 * 1024 * 128));
        classifier_.reset(new Classifier(engine, mean_file, label_file, allocator_.get()));
    }

    Classifier* TensorRTClassifier()
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
    }

    void Deactivate()
    {
    }

private:
    int device_;
    std::unique_ptr<GPUAllocator> allocator_;
    std::unique_ptr<Classifier> classifier_;
};

struct classifier_ctx
{
    ContextPool<ExecContext> pool;
};

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

            std::shared_ptr<InferenceEngine> engine(new InferenceEngine(model_file, trained_file));

            for (int i = 0; i < kContextsPerDevice; ++i)
            {
                std::unique_ptr<ExecContext> context(new ExecContext(engine, mean_file,
                                                                     label_file, dev));
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
            auto classifier = context->TensorRTClassifier();
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

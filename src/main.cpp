// realesrgan implemented with ncnn library
//
// This application performs AI-based image super-resolution (upscaling) using
// Real-ESRGAN neural network models accelerated by Vulkan GPU compute via the
// ncnn inference framework.
//
// Architecture overview:
//   The program uses a producer-consumer pipeline with three stages running
//   on separate threads:
//     1. LOAD thread  — reads and decodes input images (from files or stdin)
//     2. PROC threads — runs the Real-ESRGAN neural network inference on GPU
//     3. SAVE threads — encodes and writes the upscaled output images
//
//   Two thread-safe queues connect the stages:
//     toproc: load → proc  (priority queue, processes lowest ID first)
//     tosave: proc → save  (sequential queue, guarantees output ordering)
//
//   Multi-GPU support: multiple proc threads can be spawned across different
//   GPUs, each with its own RealESRGAN instance and tile size.

// ============================================================================
// Standard library includes
// ============================================================================
#include <stdio.h>
#include <algorithm>
#include <clocale>
#include <filesystem>
#include <iostream>
#include <map>
#include <queue>
#include <vector>
namespace fs = std::filesystem;

// ============================================================================
// Platform-specific image I/O
// ============================================================================
// On Windows, use WIC (Windows Imaging Component) for image decoding/encoding.
// On Linux/macOS, use stb_image (header-only library) for decoding and
// stb_image_write for encoding. Additionally, libpng is used for a fast
// zero-compression PNG writer optimized for stdout piping.
#if _WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else  // _WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
// libpng for fast PNG writing
#include <png.h>
#include <setjmp.h>
#include <zlib.h>
#endif  // _WIN32
#include "webp_image.h"

// ============================================================================
// Platform-specific command-line argument parsing
// ============================================================================
// Windows does not provide POSIX getopt(), so a minimal wide-character
// implementation is provided here. The parse_optarg_int_array helper parses
// comma-separated integer lists (e.g. "0,1,2") used for multi-GPU and
// tile-size arguments.
#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;

// Minimal getopt implementation for wide-character argv on Windows.
// Supports options with required arguments (indicated by ':' in optstring).
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
    if (optind >= argc || argv[optind][0] != L'-') return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL) return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc) return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

// Parse a comma-separated list of integers from a wide-character string.
// Example: L"100,200,300" → {100, 200, 300}
static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else                // _WIN32
#include <unistd.h>  // getopt()

// Parse a comma-separated list of integers from a string.
// Example: "100,200,300" → {100, 200, 300}
static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif               // _WIN32

// ============================================================================
// ncnn framework includes
// ============================================================================
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

// Real-ESRGAN model wrapper (handles tiling, padding, and inference)
#include "realesrgan.h"

// Cross-platform filesystem path utilities (path_t, PATHSTR, etc.)
#include "filesystem_utils.h"

// Print the command-line usage/help message to stderr.
static void print_usage()
{
    fprintf(stderr,
            "Usage: realesrgan-ncnn-vulkan-improved [options]...\n\n"

            "  -h                   show this help\n"

            "  -i input-path        input image path (jpg/png/webp) or "
            "directory (reads from stdin if not provided)\n"

            "  -o output-path       output image path (jpg/png/webp) or "
            "directory (outputs to stdout if not provided)\n"

            "  -s scale             upscale ratio (can be 2, 3, 4. default=4)\n"
            "  -t tile-size         tile size (>=32/0=auto, default=0) can be "
            "0,0,0 for multi-gpu\n"

            "  -m model-path        folder path to the pre-trained models. "
            "default=models\n"

            "  -n model-name        model name (default=realesr-animevideov3, "
            "can be realesr-animevideov3 | realesrgan-x4plus | "
            "realesrgan-x4plus-anime | realesrnet-x4plus)\n"

            "  -g gpu-id            gpu device to use (default=auto) can be "
            "0,1,2 for multi-gpu\n"

            "  -j load:proc:save    thread count for load/proc/save "
            "(default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n"

            "  -x                   enable tta mode\n"

            "  -f format            output image format (jpg/png/webp, "
            "default=ext/png)\n"

            "  -v                   verbose output\n");
}

// ============================================================================
// Fast PNG writer (non-Windows only)
// ============================================================================
// Uses libpng with zero compression (compression level 0) to encode PNG data
// into an in-memory buffer. This is significantly faster than stb_image_write
// for stdout piping scenarios where encoding speed matters more than file size.
#if !_WIN32

// State for the in-memory PNG writer callback. Tracks a dynamically-growing
// buffer that receives the raw PNG byte stream.
struct png_memory_writer_state
{
    unsigned char* buffer;  // Dynamically allocated output buffer
    size_t size;            // Current number of bytes written
    size_t capacity;        // Total allocated capacity of buffer
};

// libpng write callback: appends data to the in-memory buffer, growing it
// with a doubling strategy when capacity is exceeded.
static void png_write_to_memory(png_structp png_ptr,
                                png_bytep data,
                                png_size_t length)
{
    png_memory_writer_state* state =
        (png_memory_writer_state*)png_get_io_ptr(png_ptr);

    // resize buffer if needed
    if (state->size + length > state->capacity)
    {
        size_t new_capacity = state->capacity * 2;
        if (new_capacity < state->size + length)
        {
            new_capacity = state->size + length;
        }
        unsigned char* new_buffer =
            (unsigned char*)realloc(state->buffer, new_capacity);
        if (!new_buffer)
        {
            png_error(png_ptr, "Memory allocation failed");
        }
        state->buffer = new_buffer;
        state->capacity = new_capacity;
    }

    memcpy(state->buffer + state->size, data, length);
    state->size += length;
}

// libpng flush callback: no-op since we're writing to memory, not a file.
static void png_flush_memory(png_structp png_ptr)
{
    // no-op for memory writing
}

// Encode raw pixel data as a PNG image into a malloc'd memory buffer.
// Uses compression level 0 (no compression) for maximum encoding speed.
//
// Parameters:
//   data     - raw pixel data in row-major order (GRAY, RGB, or RGBA)
//   width    - image width in pixels
//   height   - image height in pixels
//   channels - number of color channels (1=gray, 3=RGB, 4=RGBA)
//   out_len  - [out] receives the size of the encoded PNG data in bytes
//
// Returns: malloc'd buffer containing the PNG data, or NULL on failure.
//          Caller is responsible for free()'ing the returned buffer.
static unsigned char* write_png_to_mem_fast(const unsigned char* data,
                                            int width,
                                            int height,
                                            int channels,
                                            int* out_len)
{
    // Create the libpng write and info structs
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return NULL;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, NULL);
        return NULL;
    }

    // libpng error handling via setjmp (standard libpng pattern)
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return NULL;
    }

    // set up memory writer with an initial capacity estimate
    png_memory_writer_state state = {0};
    state.capacity = width * height * channels + 1024;  // Initial capacity
    state.buffer = (unsigned char*)malloc(state.capacity);
    if (!state.buffer)
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return NULL;
    }

    // Redirect libpng output to our in-memory writer instead of a FILE*
    png_set_write_fn(png_ptr, &state, png_write_to_memory, png_flush_memory);

    // Map channel count to PNG color type
    int color_type;
    switch (channels)
    {
        case 1:
            color_type = PNG_COLOR_TYPE_GRAY;
            break;
        case 3:
            color_type = PNG_COLOR_TYPE_RGB;
            break;
        case 4:
            color_type = PNG_COLOR_TYPE_RGBA;
            break;
        default:
            free(state.buffer);
            png_destroy_write_struct(&png_ptr, &info_ptr);
            return NULL;
    }

    // Write the PNG header (IHDR chunk): 8-bit depth, no interlacing
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    // Disable compression entirely for maximum encoding speed.
    // The resulting file will be larger but encoding is ~10x faster.
    png_set_compression_level(png_ptr, 0);
    png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);

    png_write_info(png_ptr, info_ptr);

    // Write image data row by row
    for (int y = 0; y < height; y++)
    {
        png_write_row(png_ptr, (png_const_bytep)(data + y * width * channels));
    }

    png_write_end(png_ptr, NULL);

    *out_len = (int)state.size;
    png_destroy_write_struct(&png_ptr, &info_ptr);

    return state.buffer;
}
#endif

// ============================================================================
// Task: the unit of work that flows through the pipeline
// ============================================================================
// Each Task represents a single image being processed. It carries the input
// and output file paths, the decoded pixel data (inimage), a pre-allocated
// output buffer (outimage), and a sequential ID for ordering.
class Task
{
   public:
    int id;    // Sequential task ID (used for ordering; -233 = sentinel/poison
               // pill)
    int webp;  // Flag: 1 if the input was decoded as WebP (affects how we free
               // the pixel data)

    path_t inpath;   // Input file path (or "stdin")
    path_t outpath;  // Output file path (or "stdout")

    ncnn::Mat inimage;   // Decoded input image pixels (w × h, c channels)
    ncnn::Mat outimage;  // Pre-allocated output buffer (w*scale × h*scale, c
                         // channels)
};

// Comparator for the priority queue: tasks with lower IDs are processed first.
// This ensures that even if images are loaded out of order, the processing
// stage picks them up in sequence.
struct TaskComparator
{
    bool operator()(const Task& a, const Task& b)
    {
        return a.id > b.id;  // cigher id has lower priority
    }
};

// ============================================================================
// TaskQueue: thread-safe priority queue (load → proc)
// ============================================================================
// Used by the load thread to submit decoded images to the processing threads.
// Implements backpressure: blocks producers when the queue reaches 8 items,
// preventing unbounded memory usage from decoded images piling up.
class TaskQueue
{
   public:
    TaskQueue() {}

    // Enqueue a task. Blocks if the queue already has 8 items (backpressure).
    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 8)  // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    // Dequeue the highest-priority (lowest ID) task. Blocks if the queue is
    // empty.
    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.top();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

   private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::priority_queue<Task, std::vector<Task>, TaskComparator> tasks;
};

// ============================================================================
// SequentialTaskQueue: thread-safe ordered queue (proc → save)
// ============================================================================
// Ensures that save threads process images strictly in order (by ID).
// Processing threads may complete out of order (especially with multi-GPU),
// so this queue buffers results until the next expected ID is available.
// This guarantees that when piping to stdout, images are written in the
// same order they were read.
class SequentialTaskQueue
{
   public:
    SequentialTaskQueue() : next_id(1) {}

    // Enqueue a completed task. Blocks if the buffer has 8 items
    // (backpressure).
    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 8)  // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks[v.id] = v;

        lock.unlock();

        condition.signal();
    }

    // Dequeue the next task in sequence. Blocks until the task with
    // id == next_id is available, skipping over any that arrived early.
    void get(Task& v)
    {
        lock.lock();

        while (tasks.find(next_id) == tasks.end())
        {
            condition.wait(lock);
        }

        v = tasks[next_id];
        tasks.erase(next_id);
        next_id++;

        lock.unlock();

        condition.signal();
    }

   private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::map<int, Task> tasks;  // Buffered tasks indexed by ID
    int next_id;                // The next ID we expect to dequeue
};

// Global inter-thread queues connecting the pipeline stages
TaskQueue toproc;            // load → proc
SequentialTaskQueue tosave;  // proc → save

// ============================================================================
// stdin binary I/O helpers
// ============================================================================

// Read exactly `n` bytes from stdin into `buf`.
// Returns 1 on success, 0 if stdin is exhausted before `n` bytes are read.
static int read_bytes(unsigned char* buf, size_t n)
{
    size_t got = 0;
    while (got < n)
    {
        ssize_t r = fread(buf + got, 1, n - got, stdin);
        if (r <= 0) return 0;
        got += r;
    }
    return 1;
}

// Read a complete PNG file from stdin by parsing its chunk structure.
// PNG files consist of an 8-byte signature followed by a series of chunks,
// each with a 4-byte length, 4-byte type, variable data, and 4-byte CRC.
// This function reads chunks until it encounters the IEND (end) chunk,
// assembling the entire PNG into img_buf for later decoding by stb_image.
//
// Parameters:
//   sig_buf  - scratch buffer for the 8-byte PNG signature
//   len_buf  - scratch buffer for chunk length (4 bytes)
//   type_buf - scratch buffer for chunk type (4 bytes)
//   img_buf  - [in/out] dynamically growing buffer for the entire PNG file
//   buf_cap  - [in/out] current allocated capacity of img_buf
//   buf_len  - [in/out] current number of valid bytes in img_buf
void read_png(unsigned char* sig_buf,
              unsigned char* len_buf,
              unsigned char* type_buf,
              unsigned char*& img_buf,
              size_t& buf_cap,
              size_t& buf_len)
{
    // Expected PNG file signature (magic bytes)
    const static unsigned char png_sig[8] = {0x89, 'P',  'N',  'G',
                                             0x0D, 0x0A, 0x1A, 0x0A};

    // Read and validate the 8-byte PNG signature
    if (!read_bytes(sig_buf, 8)) return;
    if (memcmp(sig_buf, png_sig, 8))
    {
        fprintf(stderr, "Not PNG\n");
        return;
    }

    // ensure buffer can hold at least signature
    if (buf_cap < 8)
    {
        buf_cap = 8;
        unsigned char* new_buf = (unsigned char*)realloc(img_buf, buf_cap);
        if (!new_buf)
        {
            fprintf(stderr, "Failed to allocate memory for PNG buffer\n");
            return;
        }
        img_buf = new_buf;
    }
    memcpy(img_buf, sig_buf, 8);
    buf_len = 8;

    // Read PNG chunks one at a time until we hit the IEND terminator.
    // Each chunk: [4-byte length][4-byte type][length bytes data][4-byte CRC]
    for (;;)
    {
        if (!read_bytes(len_buf, 4)) return;
        if (!read_bytes(type_buf, 4)) return;
        // Decode chunk data length from big-endian 4-byte field
        uint32_t chunk_len = (len_buf[0] << 24) | (len_buf[1] << 16) |
                             (len_buf[2] << 8) | len_buf[3];

        // Sanity-check chunk length to prevent excessive allocation
        // or integer overflow (max ~100 MB per chunk)
        if (chunk_len > 0x7FFFFFFF || chunk_len > 100 * 1024 * 1024)
        {
            fprintf(stderr, "PNG chunk too large: %u bytes\n", chunk_len);
            return;
        }

        // Grow the buffer to fit: length(4) + type(4) + data(chunk_len) +
        // CRC(4)
        size_t needed = buf_len + 4 + 4 + chunk_len + 4;
        if (needed > buf_cap)
        {
            // check for potential overflow
            if (needed < buf_len)
            {
                fprintf(stderr, "PNG buffer size overflow\n");
                return;
            }

            buf_cap = needed * 1.5;
            unsigned char* new_buf = (unsigned char*)realloc(img_buf, buf_cap);
            if (!new_buf)
            {
                fprintf(stderr, "Failed to allocate memory for PNG chunk\n");
                return;
            }
            img_buf = new_buf;
        }
        // Append chunk length and type to the buffer
        memcpy(img_buf + buf_len, len_buf, 4);
        buf_len += 4;
        memcpy(img_buf + buf_len, type_buf, 4);
        buf_len += 4;

        // Append chunk data directly into the buffer from stdin
        if (!read_bytes(img_buf + buf_len, chunk_len)) return;
        buf_len += chunk_len;
        // Append the 4-byte CRC
        if (!read_bytes(img_buf + buf_len, 4)) return;
        buf_len += 4;

        // IEND marks the end of the PNG file
        if (memcmp(type_buf, "IEND", 4) == 0)
        {
            break;
        }
    }
}

// ============================================================================
// Thread parameter structures
// ============================================================================

// Parameters passed to the load thread function.
class LoadThreadParams
{
   public:
    int scale;  // Upscale factor (2, 3, or 4) — needed to pre-allocate output
                // buffer
    int jobs_load;   // Number of load jobs (currently unused within load(),
                     // always 1 thread)
    int use_stdin;   // If true, read images from stdin instead of files
    int use_stdout;  // If true, output path is set to "stdout"

    // Lists of input/output file paths (parallel arrays, same length)
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
};

// ============================================================================
// Load thread: decodes input images and submits them for processing
// ============================================================================
// Runs on a single thread. For each input image:
//   1. Reads the raw file data (from disk or stdin)
//   2. Attempts WebP decoding first, then falls back to stb_image (PNG/JPG)
//   3. Normalizes channel count (grayscale→RGB, gray+alpha→RGBA)
//   4. Wraps pixel data in an ncnn::Mat and pre-allocates the output Mat
//   5. Pushes the Task into the toproc queue
//
// When reading from stdin, the loop runs indefinitely (count increments on
// each successful decode) until stdin is exhausted (read_bytes returns 0).
void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    const int scale = ltp->scale;

    // Determine how many images to process:
    // - stdin mode: starts at 1, incremented after each successful read
    // - file mode: number of input files
    int count;
    if (ltp->use_stdin)
        count = 1;
    else
        count = ltp->input_files.size();

    // Scratch buffers for PNG stdin reading
    unsigned char sig_buf[8];
    unsigned char len_buf[4], type_buf[4];
    unsigned char* img_buf = NULL;
    size_t buf_cap = 0, buf_len = 0;

    int i = 0;
    while (i++ < count)
    {
        int webp = 0;  // Track whether this image was decoded as WebP

        unsigned char* pixeldata = 0;
        int w;  // image width
        int h;  // image height
        int c;  // number of channels (1=gray, 3=RGB, 4=RGBA)

        FILE* fp = NULL;

        // Open input file (skip if reading from stdin)
        if (!ltp->use_stdin)
        {
#if _WIN32
            fp = _wfopen(imagepath.c_str(), L"rb");
#else
            fp = fopen(ltp->input_files[i].c_str(), "rb");
#endif
        }

        if (fp)
        {
            // Read the entire file into memory for format detection and
            // decoding
            unsigned char* filedata = 0;
            int length = 0;
            {
                fseek(fp, 0, SEEK_END);
                length = ftell(fp);
                rewind(fp);
                filedata = (unsigned char*)malloc(length);
                if (filedata)
                {
                    fread(filedata, 1, length, fp);
                }
                fclose(fp);
            }

            if (filedata)
            {
                // Try WebP decoding first (webp_load returns non-NULL on
                // success)
                pixeldata = webp_load(filedata, length, &w, &h, &c);
                if (pixeldata)
                {
                    webp = 1;
                }
                else
                {
                    // Not WebP — fall back to stb_image for PNG, JPG, BMP, etc.
#if _WIN32
                    pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else   // _WIN32
                    pixeldata =
                        stbi_load_from_memory(filedata, length, &w, &h, &c, 0);
                    if (pixeldata)
                    {
                        // Normalize uncommon channel counts to standard
                        // RGB/RGBA. The neural network expects 3 or 4 channel
                        // input.
                        if (c == 1)
                        {
                            // grayscale -> rgb (reload forcing 3 channels)
                            stbi_image_free(pixeldata);
                            pixeldata = stbi_load_from_memory(filedata, length,
                                                              &w, &h, &c, 3);
                            c = 3;
                        }
                        else if (c == 2)
                        {
                            // grayscale + alpha -> rgba (reload forcing 4
                            // channels)
                            stbi_image_free(pixeldata);
                            pixeldata = stbi_load_from_memory(filedata, length,
                                                              &w, &h, &c, 4);
                            c = 4;
                        }
                    }
#endif  // _WIN32
                }

                free(filedata);
            }
        }
        // Read image from stdin (PNG format expected)
        else if (ltp->use_stdin)
        {
            // Read a complete PNG from stdin into img_buf, then decode it
            read_png(sig_buf, len_buf, type_buf, img_buf, buf_cap, buf_len);
            pixeldata = stbi_load_from_memory(img_buf, buf_len, &w, &h, &c, 0);
            if (pixeldata)
            {
                // Same channel normalization as file path above
                if (c == 1)
                {
                    // grayscale -> rgb
                    stbi_image_free(pixeldata);
                    pixeldata =
                        stbi_load_from_memory(img_buf, buf_len, &w, &h, &c, 3);
                    c = 3;
                }
                else if (c == 2)
                {
                    // grayscale + alpha -> rgba
                    stbi_image_free(pixeldata);
                    pixeldata =
                        stbi_load_from_memory(img_buf, buf_len, &w, &h, &c, 4);
                    c = 4;
                }
            }
        }

        if (pixeldata)
        {
            // Build a Task object to send through the pipeline
            Task v;
            v.id = i;
            if (ltp->use_stdin)
                v.inpath = PATHSTR("stdin");
            else
                v.inpath = ltp->input_files[i];

            if (ltp->use_stdout)
                v.outpath = PATHSTR("stdout");
            else
                v.outpath = ltp->output_files[i];

            // Wrap decoded pixels in ncnn::Mat (does NOT copy; Mat borrows the
            // pointer). Pre-allocate output Mat at the upscaled resolution.
            v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
            v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);

            // JPEG does not support alpha channels — if the image has 4
            // channels and the output format is JPEG, override the output to
            // PNG and warn.
            path_t ext = get_file_extension(v.outpath);
            if (c == 4 && (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") ||
                           ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG")))
            {
                path_t output_filename2 =
                    ltp->output_files[i] + PATHSTR(".png");
                v.outpath = output_filename2;
#if _WIN32
                fwprintf(stderr,
                         L"image %ls has alpha channel ! %ls will output %ls\n",
                         imagepath.c_str(), imagepath.c_str(),
                         output_filename2.c_str());
#else   // _WIN32
                fprintf(stderr,
                        "image %s has alpha channel ! %s will output %s\n",
                        ltp->input_files[i].c_str(),
                        ltp->input_files[i].c_str(), output_filename2.c_str());
#endif  // _WIN32
            }

            // Submit the task to processing threads
            toproc.put(v);

            // In stdin mode, free the PNG read buffer and prepare for the next
            // image. Incrementing `count` allows the loop to continue
            // indefinitely until stdin is exhausted.
            if (ltp->use_stdin)
            {
                if (img_buf)
                {
                    free(img_buf);
                    img_buf = NULL;
                }

                buf_cap = 0;
                buf_len = 0;
                count++;
            }
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"decode image %ls failed\n", imagepath.c_str());
#else   // _WIN32
            fprintf(stderr, "decode image %s failed\n",
                    ltp->input_files[i].c_str());
#endif  // _WIN32
        }
    }

    // Clean up any remaining stdin buffer
    if (img_buf)
    {
        free(img_buf);
        img_buf = NULL;
    }

    return 0;
}

// Parameters passed to each processing thread.
class ProcThreadParams
{
   public:
    const RealESRGAN*
        realesrgan;  // Pointer to the RealESRGAN instance for this GPU
};

// ============================================================================
// Proc thread: runs neural network inference on each image
// ============================================================================
// Multiple proc threads can run in parallel (one or more per GPU).
// Each thread:
//   1. Dequeues a task from `toproc`
//   2. Runs Real-ESRGAN inference (v.inimage → v.outimage)
//   3. Enqueues the result into `tosave`
//   4. Exits when it receives the poison pill (id == -233)
//
// The RealESRGAN::process() method internally handles tiling (splitting
// large images into overlapping tiles), GPU memory management, and
// neural network forward passes via ncnn's Vulkan backend.
void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const RealESRGAN* realesrgan = ptp->realesrgan;

    for (;;)
    {
        Task v;

        toproc.get(v);

        // Poison pill: signals this thread to shut down
        if (v.id == -233) break;

        // Run the super-resolution neural network
        realesrgan->process(v.inimage, v.outimage);

        // Forward the result to the save stage
        tosave.put(v);
    }

    return 0;
}

// Parameters passed to each save thread.
class SaveThreadParams
{
   public:
    int verbose;     // If true, print "input -> output done" messages
    int use_stdout;  // If true, write PNG to stdout instead of files
};

// ============================================================================
// Save thread: encodes and writes upscaled images to disk or stdout
// ============================================================================
// Multiple save threads can run in parallel for file output. For stdout mode,
// the SequentialTaskQueue guarantees images are saved in the correct order.
//
// Each thread:
//   1. Dequeues a task from `tosave` (blocks until the next sequential ID is
//   ready)
//   2. Frees the input pixel data (no longer needed after processing)
//   3. Encodes the output image in the appropriate format (PNG/JPG/WebP)
//   4. Writes to file or stdout
//   5. Exits when it receives the poison pill (id == -233)
void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;

    for (;;)
    {
        Task v;

        tosave.get(v);

        // Poison pill: signals this thread to shut down
        if (v.id == -233) break;

        // Free input pixel data — the upscaled output is in v.outimage now.
        // WebP-decoded data was allocated with malloc(), while stb_image data
        // must be freed with stbi_image_free() (which may differ on some
        // platforms).
        {
            unsigned char* pixeldata = (unsigned char*)v.inimage.data;
            if (v.webp == 1)
            {
                free(pixeldata);
            }
            else
            {
#if _WIN32
                free(pixeldata);
#else
                stbi_image_free(pixeldata);
#endif
            }
        }

        int success = 0;
        path_t ext;

        if (!stp->use_stdout)
        {
            ext = get_file_extension(v.outpath);

            // Ensure the output directory exists, creating it recursively if
            // needed
            fs::path fs_path = fs::absolute(v.outpath);
            std::string parent_path = fs_path.parent_path().string();
            if (fs::exists(parent_path) != 1)
            {
                std::cout << "Create folder: [" << parent_path << "]."
                          << std::endl;
                fs::create_directories(parent_path);
            }
        }

        // Encode and write the output image based on format
        if (stp->use_stdout)
        {
            // stdout mode: always output PNG (fastest with zero-compression
            // libpng)
            int len;
#if _WIN32
            unsigned char* png = stbi_write_png_to_mem(
                (const unsigned char*)v.outimage.data, 0, v.outimage.w,
                v.outimage.h, v.outimage.elempack, &len);
#else
            // use fast libpng implementation with no compression
            unsigned char* png = write_png_to_mem_fast(
                (const unsigned char*)v.outimage.data, v.outimage.w,
                v.outimage.h, v.outimage.elempack, &len);
#endif

            if (png != NULL)
            {
#if _WIN32

#else
                fwrite(png, 1, len, stdout);
                fflush(stdout);
#endif
#if _WIN32
                STBIW_FREE(png);
#else
                free(png);
#endif
                success = 1;
            }
        }
        else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            // WebP output encoding
            success = webp_save(v.outpath.c_str(), v.outimage.w, v.outimage.h,
                                v.outimage.elempack,
                                (const unsigned char*)v.outimage.data);
        }
        else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
            // PNG output encoding
#if _WIN32
            success =
                wic_encode_image(v.outpath.c_str(), v.outimage.w, v.outimage.h,
                                 v.outimage.elempack, v.outimage.data);
#else
            success =
                stbi_write_png(v.outpath.c_str(), v.outimage.w, v.outimage.h,
                               v.outimage.elempack, v.outimage.data, 0);
#endif
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") ||
                 ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
            // JPEG output encoding (quality=100 for maximum fidelity)
#if _WIN32
            success = wic_encode_jpeg_image(v.outpath.c_str(), v.outimage.w,
                                            v.outimage.h, v.outimage.elempack,
                                            v.outimage.data);
#else
            success =
                stbi_write_jpg(v.outpath.c_str(), v.outimage.w, v.outimage.h,
                               v.outimage.elempack, v.outimage.data, 100);
#endif
        }
        if (success)
        {
            if (verbose)
            {
#if _WIN32
                fwprintf(stderr, L"%ls -> %ls done\n", v.inpath.c_str(),
                         v.outpath.c_str());
#else
                fprintf(stderr, "%s -> %s done\n", v.inpath.c_str(),
                        v.outpath.c_str());
#endif
            }
        }
        else
        {
#if _WIN32
            fwprintf(stderr, L"encode image %ls failed\n", v.outpath.c_str());
#else
            fprintf(stderr, "encode image %s failed\n", v.outpath.c_str());
#endif
        }
    }

    return 0;
}

// ============================================================================
// Main entry point
// ============================================================================
// Orchestrates the entire upscaling pipeline:
//   1. Parse command-line arguments
//   2. Validate inputs and resolve file paths
//   3. Initialize Vulkan GPU instance(s) and RealESRGAN model(s)
//   4. Launch load, proc, and save threads
//   5. Wait for completion and clean up
#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    // ---- Default configuration values ----
    path_t inputpath;
    path_t outputpath;
    int scale = 4;                     // Default upscale factor
    std::vector<int> tilesize;         // Per-GPU tile sizes (0 = auto)
    path_t model = PATHSTR("models");  // Directory containing model files
    path_t modelname =
        PATHSTR("realesr-animevideov3");  // Default model (anime-optimized)
    std::vector<int> gpuid;               // GPU device IDs to use
    int jobs_load = 1;                    // Number of image loading threads
    std::vector<int> jobs_proc;           // Per-GPU processing thread counts
    int jobs_save = 2;                    // Number of image saving threads
    int verbose = 0;                      // Verbose logging flag
    int tta_mode =
        0;  // Test-Time Augmentation (8x slower, slightly better quality)
    path_t format = PATHSTR("png");  // Default output format

    // ---- Parse command-line arguments ----
#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:o:s:t:m:n:g:j:f:vxh")) != (wchar_t)-1)
    {
        switch (opt)
        {
            case L'i':
                inputpath = optarg;
                break;
            case L'o':
                outputpath = optarg;
                break;
            case L's':
                scale = _wtoi(optarg);
                break;
            case L't':
                tilesize = parse_optarg_int_array(optarg);
                break;
            case L'm':
                model = optarg;
                break;
            case L'n':
                modelname = optarg;
                break;
            case L'g':
                gpuid = parse_optarg_int_array(optarg);
                break;
            case L'j':
                // Parse "load:proc:save" thread counts.
                // The proc part can be comma-separated for multi-GPU (e.g.
                // "1:2,2,2:2").
                swscanf(optarg, L"%d:%*[^:]:%d", &jobs_load, &jobs_save);
                jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
                break;
            case L'f':
                format = optarg;
                break;
            case L'v':
                verbose = 1;
                break;
            case L'x':
                tta_mode = 1;
                break;
            case L'h':
            default:
                print_usage();
                return -1;
        }
    }
#else   // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "i:o:s:t:m:n:g:j:f:vxh")) != -1)
    {
        switch (opt)
        {
            case 'i':
                inputpath = optarg;
                break;
            case 'o':
                outputpath = optarg;
                break;
            case 's':
                scale = atoi(optarg);
                break;
            case 't':
                tilesize = parse_optarg_int_array(optarg);
                break;
            case 'm':
                model = optarg;
                break;
            case 'n':
                modelname = optarg;
                break;
            case 'g':
                gpuid = parse_optarg_int_array(optarg);
                break;
            case 'j':
                // Parse "load:proc:save" thread counts.
                // The proc part can be comma-separated for multi-GPU (e.g.
                // "1:2,2,2:2").
                sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
                jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
                break;
            case 'f':
                format = optarg;
                break;
            case 'v':
                verbose = 1;
                break;
            case 'x':
                tta_mode = 1;
                break;
            case 'h':
            default:
                print_usage();
                return -1;
        }
    }
#endif  // _WIN32

    // ---- Configure stdin/stdout mode when paths are omitted ----
    if (inputpath.empty())
    {
        fprintf(stderr, "using stdin as input\n");
    }

    if (outputpath.empty())
    {
        fprintf(stderr, "using stdout as output\n");
        // Disable stb PNG compression for faster stdout writing
        stbi_write_png_compression_level = 0;
    }

    // ---- Validate arguments ----

    // Tile size count must match GPU count (one tile size per GPU)
    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) &&
        !tilesize.empty())
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    // Tile size must be 0 (auto) or at least 32 pixels
    for (int i = 0; i < (int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "invalid tilesize argument\n");
            return -1;
        }
    }

    // Thread counts must be positive
    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    // Processing thread count must match GPU count
    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) &&
        !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i = 0; i < (int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }

    // ---- Determine output format ----
    // When a single output file is specified (not a directory), infer the
    // format from the file extension, ignoring the -f argument.
    if (!path_is_directory(outputpath) && !outputpath.empty())
    {
        // guess format from outputpath no matter what format argument specified
        path_t ext = get_file_extension(outputpath);

        if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
            format = PATHSTR("png");
        }
        else if (ext == PATHSTR("webp") || ext == PATHSTR("WEBP"))
        {
            format = PATHSTR("webp");
        }
        else if (ext == PATHSTR("jpg") || ext == PATHSTR("JPG") ||
                 ext == PATHSTR("jpeg") || ext == PATHSTR("JPEG"))
        {
            format = PATHSTR("jpg");
        }
        else
        {
            fprintf(stderr, "invalid outputpath extension type\n");
            return -1;
        }
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") &&
        format != PATHSTR("jpg"))
    {
        fprintf(stderr, "invalid format argument\n");
        return -1;
    }

    // ---- Collect input and output file paths ----
    // Supports two modes:
    //   1. Directory→Directory: processes all images in the input directory
    //   2. File→File: processes a single image
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
    {
        if (path_is_directory(inputpath) && path_is_directory(outputpath))
        {
            // Batch mode: list all files in input directory
            std::vector<path_t> filenames;
            int lr = list_directory(inputpath, filenames);
            if (lr != 0) return -1;

            const int count = filenames.size();
            input_files.resize(count);
            output_files.resize(count);

            // Track previous filename to detect collisions (e.g. foo.png and
            // foo.jpg would both produce foo.png output). When detected, append
            // the original extension to disambiguate (e.g. foo.jpg.png).
            path_t last_filename;
            path_t last_filename_noext;
            for (int i = 0; i < count; i++)
            {
                path_t filename = filenames[i];
                path_t filename_noext =
                    get_file_name_without_extension(filename);
                path_t output_filename = filename_noext + PATHSTR('.') + format;

                // filename list is sorted, check if output image path conflicts
                if (filename_noext == last_filename_noext)
                {
                    path_t output_filename2 = filename + PATHSTR('.') + format;
#if _WIN32
                    fwprintf(
                        stderr,
                        L"both %ls and %ls output %ls ! %ls will output %ls\n",
                        filename.c_str(), last_filename.c_str(),
                        output_filename.c_str(), filename.c_str(),
                        output_filename2.c_str());
#else
                    fprintf(stderr,
                            "both %s and %s output %s ! %s will output %s\n",
                            filename.c_str(), last_filename.c_str(),
                            output_filename.c_str(), filename.c_str(),
                            output_filename2.c_str());
#endif
                    output_filename = output_filename2;
                }
                else
                {
                    last_filename = filename;
                    last_filename_noext = filename_noext;
                }

                input_files[i] = inputpath + PATHSTR('/') + filename;
                output_files[i] = outputpath + PATHSTR('/') + output_filename;
            }
        }
        else if (!path_is_directory(inputpath) &&
                 !path_is_directory(outputpath))
        {
            // Single-file mode
            input_files.push_back(inputpath);
            output_files.push_back(outputpath);
        }
        else
        {
            fprintf(stderr,
                    "inputpath and outputpath must be either file or directory "
                    "at the same time\n");
            return -1;
        }
    }

    // ---- Set model-specific pre-padding ----
    // Pre-padding is extra border pixels added around each tile before
    // inference to avoid edge artifacts from the neural network's receptive
    // field.
    int prepadding = 0;

    if (model.find(PATHSTR("models")) != path_t::npos ||
        model.find(PATHSTR("models2")) != path_t::npos)
    {
        prepadding = 10;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    // Previously used for model name validation; commented out to allow
    // custom model names without restriction.
    // if (modelname.find(PATHSTR("realesrgan-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("realesrnet-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("esrgan-x4")) != path_t::npos)
    // {}
    // else
    // {
    //     fprintf(stderr, "unknown model name\n");
    //     return -1;
    // }

    // ---- Construct model file paths ----
    // The animevideov3 model has scale-specific weights (e.g.
    // realesr-animevideov3-x2.bin), while other models have a single weight
    // file for all scales.
#if _WIN32
    wchar_t parampath[256];
    wchar_t modelpath[256];

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        swprintf(parampath, 256, L"%s/%s-x%s.param", model.c_str(),
                 modelname.c_str(), std::to_string(scale));
        swprintf(modelpath, 256, L"%s/%s-x%s.bin", model.c_str(),
                 modelname.c_str(), std::to_string(scale));
    }
    else
    {
        swprintf(parampath, 256, L"%s/%s.param", model.c_str(),
                 modelname.c_str());
        swprintf(modelpath, 256, L"%s/%s.bin", model.c_str(),
                 modelname.c_str());
    }

#else
    char parampath[256];
    char modelpath[256];

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        sprintf(parampath, "%s/%s-x%s.param", model.c_str(), modelname.c_str(),
                std::to_string(scale).c_str());
        sprintf(modelpath, "%s/%s-x%s.bin", model.c_str(), modelname.c_str(),
                std::to_string(scale).c_str());
    }
    else
    {
        sprintf(parampath, "%s/%s.param", model.c_str(), modelname.c_str());
        sprintf(modelpath, "%s/%s.bin", model.c_str(), modelname.c_str());
    }
#endif

    // Resolve relative paths and normalize separators
    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

    // ---- Initialize Vulkan GPU runtime ----
#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    // Default to the first available GPU if none specified
    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    // Default to 2 processing threads per GPU if not specified
    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    // Default tile size to 0 (auto) for each GPU if not specified
    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    // Cap load/save thread counts to the number of CPU cores available
    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    // stdin/stdout modes are inherently single-threaded (serial I/O)
    if (inputpath.empty()) jobs_load = 1;
    if (outputpath.empty()) jobs_save = 1;

    // Validate that all requested GPU IDs exist
    int gpu_count = ncnn::get_gpu_count();
    for (int i = 0; i < use_gpu_count; i++)
    {
        if (gpuid[i] < 0 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    // Cap processing threads per GPU to the GPU's available compute queues.
    // Having more threads than queues provides no benefit and wastes resources.
    int total_jobs_proc = 0;
    for (int i = 0; i < use_gpu_count; i++)
    {
        int gpu_queue_count =
            ncnn::get_gpu_info(gpuid[i]).compute_queue_count();
        jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
        total_jobs_proc += jobs_proc[i];
    }

    // ---- Auto-detect tile size based on GPU VRAM budget ----
    // Larger tiles are more efficient but require more VRAM. This heuristic
    // selects the largest tile size that fits in the available memory.
    for (int i = 0; i < use_gpu_count; i++)
    {
        if (tilesize[i] != 0)
            continue;  // User specified a tile size, skip auto

        uint32_t heap_budget =
            ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

        // more fine-grained tilesize policy here
        if (model.find(PATHSTR("models")) != path_t::npos)
        {
            if (heap_budget > 1900)
                tilesize[i] = 200;
            else if (heap_budget > 550)
                tilesize[i] = 100;
            else if (heap_budget > 190)
                tilesize[i] = 64;
            else
                tilesize[i] = 32;
        }
    }

    // ============================================================================
    // Main processing pipeline
    // ============================================================================
    {
        // Create one RealESRGAN instance per GPU, each loading the same model
        // but configured with GPU-specific tile size and device ID.
        std::vector<RealESRGAN*> realesrgan(use_gpu_count);

        for (int i = 0; i < use_gpu_count; i++)
        {
            realesrgan[i] = new RealESRGAN(gpuid[i], tta_mode);

            realesrgan[i]->load(paramfullpath, modelfullpath);

            realesrgan[i]->scale = scale;
            realesrgan[i]->tilesize = tilesize[i];
            realesrgan[i]->prepadding = prepadding;
        }

        // ---- Launch the three-stage thread pipeline ----
        {
            // Stage 1: Image loading thread
            LoadThreadParams ltp;
            ltp.scale = scale;
            ltp.jobs_load = jobs_load;
            ltp.input_files = input_files;
            ltp.output_files = output_files;

            if (inputpath.empty())
                ltp.use_stdin = 1;
            else
                ltp.use_stdin = 0;

            if (outputpath.empty())
                ltp.use_stdout = 1;
            else
                ltp.use_stdout = 0;

            ncnn::Thread load_thread(load, (void*)&ltp);

            // Stage 2: GPU processing threads (one or more per GPU)
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i = 0; i < use_gpu_count; i++)
            {
                ptp[i].realesrgan = realesrgan[i];
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i = 0; i < use_gpu_count; i++)
                {
                    for (int j = 0; j < jobs_proc[i]; j++)
                    {
                        proc_threads[total_jobs_proc_id++] =
                            new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                }
            }

            // Stage 3: Image saving threads
            SaveThreadParams stp;
            stp.verbose = verbose;
            if (outputpath.empty())
                stp.use_stdout = 1;
            else
                stp.use_stdout = 0;

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i = 0; i < jobs_save; i++)
            {
                save_threads[i] = new ncnn::Thread(save, (void*)&stp);
            }

            // ---- Graceful shutdown sequence ----

            // Wait for the load thread to finish reading all input images
            load_thread.join();

            // Send poison pills (id == -233) to all proc threads to signal
            // shutdown. One poison pill per proc thread ensures each thread
            // receives exactly one.
            Task end;
            end.id = -233;

            for (int i = 0; i < total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            // Wait for all proc threads to finish processing and shut down
            for (int i = 0; i < total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            // Send poison pills to all save threads (proc is done, so all
            // results have been forwarded to tosave by now)
            for (int i = 0; i < jobs_save; i++)
            {
                tosave.put(end);
            }

            // Wait for all save threads to finish writing output images
            for (int i = 0; i < jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }
        }

        // Clean up RealESRGAN model instances (releases GPU resources)
        for (int i = 0; i < use_gpu_count; i++)
        {
            delete realesrgan[i];
        }
        realesrgan.clear();
    }

    // Tear down the Vulkan GPU runtime
    ncnn::destroy_gpu_instance();

    return 0;
}

// realesrgan implemented with ncnn library
#include <stdio.h>
#include <algorithm>
#include <clocale>
#include <filesystem>
#include <iostream>
#include <queue>
#include <vector>
namespace fs = std::filesystem;

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

#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
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

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "realesrgan.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr,
            "Usage: realesrgan-ncnn-vulkan [options]...\n\n"

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

#if !_WIN32
// fast PNG writer using libpng with no compression
struct png_memory_writer_state
{
    unsigned char* buffer;
    size_t size;
    size_t capacity;
};

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

static void png_flush_memory(png_structp png_ptr)
{
    // no-op for memory writing
}

static unsigned char* write_png_to_mem_fast(const unsigned char* data,
                                            int width,
                                            int height,
                                            int channels,
                                            int* out_len)
{
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return NULL;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, NULL);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return NULL;
    }

    // set up memory writer
    png_memory_writer_state state = {0};
    state.capacity = width * height * channels + 1024;  // Initial capacity
    state.buffer = (unsigned char*)malloc(state.capacity);
    if (!state.buffer)
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return NULL;
    }

    png_set_write_fn(png_ptr, &state, png_write_to_memory, png_flush_memory);

    // set PNG parameters for maximum speed (no compression)
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

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    // set compression level to 0 for maximum speed
    png_set_compression_level(png_ptr, 0);
    png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);

    png_write_info(png_ptr, info_ptr);

    // write image data
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

class Task
{
   public:
    int id;
    int webp;

    path_t inpath;
    path_t outpath;

    ncnn::Mat inimage;
    ncnn::Mat outimage;
};

class TaskQueue
{
   public:
    TaskQueue() {}

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

    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

   private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;

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

void read_png(unsigned char* sig_buf,
              unsigned char* len_buf,
              unsigned char* type_buf,
              unsigned char*& img_buf,
              size_t& buf_cap,
              size_t& buf_len)
{
    const static unsigned char png_sig[8] = {0x89, 'P',  'N',  'G',
                                             0x0D, 0x0A, 0x1A, 0x0A};

    // signature
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

    // read chunks until IEND
    for (;;)
    {
        if (!read_bytes(len_buf, 4)) return;
        if (!read_bytes(type_buf, 4)) return;
        // chunk length (big-endian)
        uint32_t chunk_len = (len_buf[0] << 24) | (len_buf[1] << 16) |
                             (len_buf[2] << 8) | len_buf[3];

        // validate chunk length to prevent overflow and excessive allocation
        if (chunk_len > 0x7FFFFFFF || chunk_len > 100 * 1024 * 1024)
        {
            fprintf(stderr, "PNG chunk too large: %u bytes\n", chunk_len);
            return;
        }

        // ensure capacity
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
        // copy length+type
        memcpy(img_buf + buf_len, len_buf, 4);
        buf_len += 4;
        memcpy(img_buf + buf_len, type_buf, 4);
        buf_len += 4;

        // copy data
        if (!read_bytes(img_buf + buf_len, chunk_len)) return;
        buf_len += chunk_len;
        // copy CRC
        if (!read_bytes(img_buf + buf_len, 4)) return;
        buf_len += 4;

        // check for IEND
        if (memcmp(type_buf, "IEND", 4) == 0)
        {
            break;
        }
    }
}

class LoadThreadParams
{
   public:
    int scale;
    int jobs_load;
    int use_stdin;
    int use_stdout;

    // session data
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    const int scale = ltp->scale;

    int count;
    if (ltp->use_stdin)
        count = 1;
    else
        count = ltp->input_files.size();

    unsigned char sig_buf[8];
    unsigned char len_buf[4], type_buf[4];
    unsigned char* img_buf = NULL;
    size_t buf_cap = 0, buf_len = 0;

    int i = 0;
    while (i++ < count)
    {
        int webp = 0;

        unsigned char* pixeldata = 0;
        int w;
        int h;
        int c;

        FILE* fp = NULL;

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
            // read whole file
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
                pixeldata = webp_load(filedata, length, &w, &h, &c);
                if (pixeldata)
                {
                    webp = 1;
                }
                else
                {
                    // not webp, try jpg png etc.
#if _WIN32
                    pixeldata = wic_decode_image(imagepath.c_str(), &w, &h, &c);
#else   // _WIN32
                    pixeldata =
                        stbi_load_from_memory(filedata, length, &w, &h, &c, 0);
                    if (pixeldata)
                    {
                        // stb_image auto channel
                        if (c == 1)
                        {
                            // grayscale -> rgb
                            stbi_image_free(pixeldata);
                            pixeldata = stbi_load_from_memory(filedata, length,
                                                              &w, &h, &c, 3);
                            c = 3;
                        }
                        else if (c == 2)
                        {
                            // grayscale + alpha -> rgba
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
        // read from stdin
        else if (ltp->use_stdin)
        {
            read_png(sig_buf, len_buf, type_buf, img_buf, buf_cap, buf_len);
            pixeldata = stbi_load_from_memory(img_buf, buf_len, &w, &h, &c, 0);
            if (pixeldata)
            {
                // stb_image auto channel
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

            v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
            v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);

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

            toproc.put(v);

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

    if (img_buf)
    {
        free(img_buf);
        img_buf = NULL;
    }

    return 0;
}

class ProcThreadParams
{
   public:
    const RealESRGAN* realesrgan;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const RealESRGAN* realesrgan = ptp->realesrgan;

    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233) break;

        realesrgan->process(v.inimage, v.outimage);

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams
{
   public:
    int verbose;
    int use_stdout;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;

    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233) break;

        // free input pixel data
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

            /* ----------- Create folder if not exists -------------------*/
            fs::path fs_path = fs::absolute(v.outpath);
            std::string parent_path = fs_path.parent_path().string();
            if (fs::exists(parent_path) != 1)
            {
                std::cout << "Create folder: [" << parent_path << "]."
                          << std::endl;
                fs::create_directories(parent_path);
            }
        }

        if (stp->use_stdout)
        {
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
            success = webp_save(v.outpath.c_str(), v.outimage.w, v.outimage.h,
                                v.outimage.elempack,
                                (const unsigned char*)v.outimage.data);
        }
        else if (ext == PATHSTR("png") || ext == PATHSTR("PNG"))
        {
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

#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    path_t inputpath;
    path_t outputpath;
    int scale = 4;
    std::vector<int> tilesize;
    path_t model = PATHSTR("models");
    path_t modelname = PATHSTR("realesr-animevideov3");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    path_t format = PATHSTR("png");

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

    if (inputpath.empty())
    {
        fprintf(stderr, "using stdin as input\n");
    }

    if (outputpath.empty())
    {
        fprintf(stderr, "using stdout as output\n");
        stbi_write_png_compression_level = 0;
    }

    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) &&
        !tilesize.empty())
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    for (int i = 0; i < (int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "invalid tilesize argument\n");
            return -1;
        }
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

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

    // collect input and output filepath
    std::vector<path_t> input_files;
    std::vector<path_t> output_files;
    {
        if (path_is_directory(inputpath) && path_is_directory(outputpath))
        {
            std::vector<path_t> filenames;
            int lr = list_directory(inputpath, filenames);
            if (lr != 0) return -1;

            const int count = filenames.size();
            input_files.resize(count);
            output_files.resize(count);

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

    // if (modelname.find(PATHSTR("realesrgan-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("realesrnet-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("esrgan-x4")) != path_t::npos)
    // {}
    // else
    // {
    //     fprintf(stderr, "unknown model name\n");
    //     return -1;
    // }

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

    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    if (inputpath.empty()) jobs_load = 1;
    if (outputpath.empty()) jobs_save = 1;

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

    int total_jobs_proc = 0;
    for (int i = 0; i < use_gpu_count; i++)
    {
        int gpu_queue_count =
            ncnn::get_gpu_info(gpuid[i]).compute_queue_count();
        jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
        total_jobs_proc += jobs_proc[i];
    }

    for (int i = 0; i < use_gpu_count; i++)
    {
        if (tilesize[i] != 0) continue;

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

    {
        std::vector<RealESRGAN*> realesrgan(use_gpu_count);

        for (int i = 0; i < use_gpu_count; i++)
        {
            realesrgan[i] = new RealESRGAN(gpuid[i], tta_mode);

            realesrgan[i]->load(paramfullpath, modelfullpath);

            realesrgan[i]->scale = scale;
            realesrgan[i]->tilesize = tilesize[i];
            realesrgan[i]->prepadding = prepadding;
        }

        // main routine
        {
            // load image
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

            // realesrgan proc
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

            // save image
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

            // end
            load_thread.join();

            Task end;
            end.id = -233;

            for (int i = 0; i < total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i = 0; i < total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i = 0; i < jobs_save; i++)
            {
                tosave.put(end);
            }

            for (int i = 0; i < jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }
        }

        for (int i = 0; i < use_gpu_count; i++)
        {
            delete realesrgan[i];
        }
        realesrgan.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}

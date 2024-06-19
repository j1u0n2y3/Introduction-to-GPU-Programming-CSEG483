#pragma once
#include "FreeImage/FreeImage.h"

class IO_Images {
public:
    unsigned int width, height, pitch;
    size_t data_bytes;
    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* data;
    } input;
    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* data;
    } output;
    ~IO_Images() {
        FreeImage_Unload(input.fi_bitmap_32);
        FreeImage_Unload(output.fi_bitmap_32);
        delete[] output.data;
    }
};

#define GAUSSIAN_FILTER_5  0
const float filter_5[1][25] = {
    {
        0.0037650,  0.0150190,  0.0237920,  0.0150190,  0.0037650,
        0.0150190,  0.0599121,  0.0949072,  0.0599121,  0.0150190,
        0.0237920,  0.0949072,  0.1503423,  0.0949072,  0.0237920,
        0.0150190,  0.0599121,  0.0949072,  0.0599121,  0.0150190,
        0.0037650,  0.0150190,  0.0237920,  0.0150190,  0.0037650
    }
};

void read_input_image_into_RGBA_image(IO_Images& io_images, const char* filename);
void prepare_output_image(IO_Images& io_images);
void write_filtered_data_into_output_image(IO_Images& io_images, const char* filename);

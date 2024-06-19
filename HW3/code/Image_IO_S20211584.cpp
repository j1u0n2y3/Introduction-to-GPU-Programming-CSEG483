#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Image_IO.h"

//#define DEBUG_LOGGING

void read_input_image_into_RGBA_image(IO_Images &io_images, const char *filename)
{
	// Assume everything is fine with reading image from input image file: no error checking is done.
	FREE_IMAGE_FORMAT image_format = FreeImage_GetFileType(filename, 0);
	FIBITMAP *fi_bitmap = FreeImage_Load(image_format, filename);

	io_images.width = FreeImage_GetWidth(fi_bitmap);
	io_images.height = FreeImage_GetHeight(fi_bitmap);
	int bits_per_pixel = FreeImage_GetBPP(fi_bitmap);
#ifdef DEBUG_LOGGING
	fprintf(stdout, "*** Reading a %d-bit image of %d x %d pixels from %s.\n", bits_per_pixel,
			io_images.width, io_images.height, filename);
#endif
	FIBITMAP *fi_bitmap_32;
	if (bits_per_pixel == 32)
	{
		io_images.input.fi_bitmap_32 = fi_bitmap;
	}
	else
	{
#ifdef DEBUG_LOGGING
		fprintf(stdout, "   - Converting texture from %d bits to 32 bits...\n", bits_per_pixel);
#endif
		io_images.input.fi_bitmap_32 = FreeImage_ConvertTo32Bits(fi_bitmap);
		FreeImage_Unload(fi_bitmap);
	}

	io_images.pitch = FreeImage_GetPitch(io_images.input.fi_bitmap_32);
#ifdef DEBUG_LOGGING
	fprintf(stdout, "   - input image: width = %d, height = %d, bpp = %d, pitch = %d\n", io_images.width,
			io_images.height, FreeImage_GetBPP(io_images.input.fi_bitmap_32), io_images.pitch);
#endif
	if (io_images.width * 4 == io_images.pitch)
	{
		io_images.data_bytes = io_images.pitch * io_images.height * sizeof(unsigned char);
		io_images.input.data = FreeImage_GetBits(io_images.input.fi_bitmap_32);
	}
	else
	{
#ifdef DEBUG_LOGGING
		fprintf(stderr, "*** Error: do more to handle the case of pitch being different from width...\n");
#endif
		exit(-1);
	}
#ifdef DEBUG_LOGGING
	fprintf(stdout, "*** Done!\n");
#endif
}

void prepare_output_image(IO_Images &io_images)
{
	io_images.output.data = new unsigned char[io_images.data_bytes];
	if (io_images.output.data == NULL)
	{
		fprintf(stderr, "*** Error: cannot allocate memory for output image...\n");
		exit(-1);
	}
}

void write_filtered_data_into_output_image(IO_Images &io_images, const char *filename)
{
	// Assume everything is fine with writing output image into file: no error checking is done.
	// assume width == pitch
#ifdef DEBUG_LOGGING
	fprintf(stdout, "\n*** Writing a 32-bit image of %d x %d pixels into %s.\n",
			io_images.width, io_images.height, filename);
#endif
	io_images.output.fi_bitmap_32 = FreeImage_ConvertFromRawBits(io_images.output.data,
																 io_images.width, io_images.height, io_images.pitch, 32,
																 FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, FALSE);
	FreeImage_Save(FIF_PNG, io_images.output.fi_bitmap_32, filename, 0);
#ifdef DEBUG_LOGGING
	fprintf(stdout, "*** Done!\n");
#endif
}